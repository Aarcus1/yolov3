import os
import torch
import re
from typing import Dict, Any, Optional, List
from source.serialization.checkpoint_mode import CheckpointMode
from source.utilities.metrics import NameMetricPair
from source.utilities.utilities import (
    get_device,
    get_detection_model,
    create_optimizer,
    create_combined_scheduler,
)

_EPOCH_REGEX = re.compile(r"_epo(\d+)\.pth$")


def build_model_and_optimizer(main_config: Dict[str, Any], device: torch.device, train_steps_per_epoch: int, ):
    model = get_detection_model(
        num_classes=main_config["dataset"]["class_count"],
        backbone_config_path=main_config["model"]["backbone_config_path"],
        device=device,
    )

    optimizer = create_optimizer(
        model,
        main_config["training"]["optimizer"],
    )

    scaler = torch.amp.GradScaler(device.type, enabled=main_config["training"]["use_amp"] and device.type == "cuda")

    scheduler = create_combined_scheduler(
        optimizer,
        scheduler_config=main_config["training"]["scheduler"],
        warmup_config=main_config["training"]["warmup"],
        train_steps_per_epoch=train_steps_per_epoch,
        total_epochs=main_config["training"]["epochs"],
    )

    return model, optimizer, scaler, scheduler


def init_from_checkpoint(
        checkpoint_path: str,
        main_config: Dict[str, Any],
        device: torch.device,
        train_steps_per_epoch: int,
):
    print(f"[INIT] Resuming training from checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model, optimizer, scaler, scheduler = build_model_and_optimizer(main_config, device, train_steps_per_epoch)

    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    if "scaler" in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = checkpoint["epoch"] + 1

    del checkpoint
    torch.cuda.empty_cache()

    return {
        "model": model,
        "optimizer": optimizer,
        "scaler": scaler,
        "scheduler": scheduler,
        "start_epoch": start_epoch,
    }


def init_from_scratch(
        main_config: Dict[str, Any],
        device: torch.device,
        train_steps_per_epoch: int,
):
    print("[INIT] Training from scratch")

    model, optimizer, scaler, scheduler = build_model_and_optimizer(main_config, device, train_steps_per_epoch)
    backbone_config = main_config["training"]["checkpoints"]["backbone"]
    if backbone_config["enabled"]:
        weight_path = backbone_config["weight_path"]
        model.backbone.load_weights(weight_path)
    return {
        "model": model,
        "optimizer": optimizer,
        "scaler": scaler,
        "scheduler": scheduler,
        "start_epoch": 1,
    }


def find_latest_checkpoint(
        checkpoint_dir: str,
        snapshot_path_base: str,
        train_steps_per_epoch: int,
) -> Optional[str]:
    if not os.path.isdir(checkpoint_dir):
        return None

    best_epoch = -1
    best_path: Optional[str] = None
    base_name = os.path.basename(snapshot_path_base)

    for file_name in os.listdir(checkpoint_dir):
        if not file_name.startswith(base_name):
            continue

        match = _EPOCH_REGEX.search(file_name)
        if match is None:
            continue

        epoch = int(match.group(1))
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = os.path.join(checkpoint_dir, file_name)

    if best_path is not None:
        print(
            f"[CHECKPOINT] Found latest checkpoint "
            f"(epoch={best_epoch}): {best_path}"
        )

    return best_path


def initialize_training(main_config: Dict[str, Any], device: torch.device, snapshot_path_base: str,
                        train_steps_per_epoch: int):
    checkpoint_config = main_config["training"]["checkpoints"]
    checkpoint_type = CheckpointMode(checkpoint_config["mode"])

    if checkpoint_type == CheckpointMode.LATEST:
        checkpoint_directory = checkpoint_config["directory_path"]
        maybe_latest_checkpoint = find_latest_checkpoint(checkpoint_directory, snapshot_path_base,
                                                         train_steps_per_epoch)

        if maybe_latest_checkpoint is None:
            print("[INIT] No checkpoint found, falling back to scratch")
            return init_from_scratch(main_config, device, train_steps_per_epoch)

        return init_from_checkpoint(maybe_latest_checkpoint, main_config, device, train_steps_per_epoch)

    if checkpoint_type == CheckpointMode.CUSTOM:
        custom_path = checkpoint_config["custom_path"]
        return init_from_checkpoint(custom_path, main_config, device, train_steps_per_epoch)

    return init_from_scratch(main_config, device, train_steps_per_epoch)


def initialize_testing(main_config: Dict[str, Any], device: torch.device, snapshot_path_base: str,
                       train_steps_per_epoch: int):
    checkpoint_config = main_config["testing"]["checkpoints"]
    checkpoint_type = CheckpointMode(checkpoint_config["mode"])

    if checkpoint_type == CheckpointMode.LATEST:
        checkpoint_directory = checkpoint_config["directory_path"]
        maybe_latest_checkpoint = find_latest_checkpoint(checkpoint_directory, snapshot_path_base,
                                                         train_steps_per_epoch)

        if maybe_latest_checkpoint is None:
            raise RuntimeError("[TESTING INIT] No checkpoint found for testing.")

        return init_from_checkpoint(maybe_latest_checkpoint, main_config, device, train_steps_per_epoch)

    if checkpoint_type == CheckpointMode.CUSTOM:
        custom_path = checkpoint_config["custom_path"]
        return init_from_checkpoint(custom_path, main_config, device, train_steps_per_epoch)

    raise RuntimeError("[TESTING INIT] Unsupported checkpoint mode for testing.")


def save_checkpoint(state: dict, checkpoint_path: str, epoch: int):
    dirpath = os.path.dirname(checkpoint_path)
    os.makedirs(dirpath, exist_ok=True)

    if os.path.isdir(checkpoint_path):
        raise RuntimeError(
            f"Checkpoint path exists as a directory: {checkpoint_path}"
        )

    torch.save(
        {
            **state,
            "epoch": epoch,
        },
        checkpoint_path,
    )


def get_snapshot_path(main_config: Dict[str, Any]) -> str:
    path_parts = []

    model_path = main_config["training"]["checkpoints"]["directory_path"]

    path_parts.append(os.path.join(model_path, "YOLO"))

    path_parts[-1] += f"_cls{main_config['dataset']['class_count']}"
    path_parts[-1] += f"_bs{main_config['dataset']['training']['batch_size']}"

    lr = main_config["training"]["optimizer"]["lr"]
    path_parts[-1] += f"_lr{lr:.0e}"

    path_parts[-1] += f"_{main_config['training']['optimizer']['type']}"

    img_size = main_config["dataset"]["image_size"]
    if isinstance(img_size, int):
        path_parts[-1] += f"_{img_size}"
    else:
        path_parts[-1] += f"_{img_size[0]}x{img_size[1]}"

    path_parts[-1] += f"_epo{main_config['training']['epochs']}"
    path_parts[-1] += f"_s{main_config['general']['determinism']['seed']}"

    snapshot_path = "".join(path_parts)

    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)

    return snapshot_path


def save_metrics_to_txt(metrics: List[NameMetricPair], filename: str = "metrics.txt") -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for name, value in metrics:
            f.write(f"{name}: {value}\n")
