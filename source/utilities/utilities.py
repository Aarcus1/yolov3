import random
from typing import List, Dict, Any
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ChainedScheduler

import numpy as np
import torch
import yaml
import os
import torch.nn as nn
from source.model.yolo import YOLOv3
from enum import Enum


def load_main_config(args):
    with open(os.path.join(args.config_folder, 'main_config.yaml'), "r") as file:
        main_config = yaml.safe_load(file)
    return main_config


class DeterminismMode(Enum):
    NONE = "none"  # Fastest, no reproducibility guarantees
    SAFE = "safe"  # Deterministic seeding only (no runtime penalty)
    FULL = "full"  # Full determinism (may reduce performance)


def set_torch_determinism(determinism_config: dict):
    seed = determinism_config['seed']
    mode = DeterminismMode(determinism_config['mode'])

    if mode is DeterminismMode.NONE:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if mode is DeterminismMode.FULL:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        torch.use_deterministic_algorithms(False)


def get_device(device_str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    return torch.device(device_str)


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    optimizer_type = config['type'].lower()
    lr = config['lr']
    weight_decay = config['weight_decay']

    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        momentum = config['momentum']
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def create_combined_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Dict[str, Any],
    warmup_config: Dict[str, Any],
    train_steps_per_epoch: int,
    total_epochs: int
):
    base_lr = optimizer.param_groups[0]['lr']

    _enabled = warmup_config["enabled"]
    warmup_enabled = _enabled if isinstance(_enabled, bool) else str(_enabled).lower() == "true"
    warmup_epochs = int(warmup_config["epochs"])
    start_lr = float(warmup_config["start_lr"])

    total_warmup_steps = warmup_epochs * train_steps_per_epoch

    if warmup_enabled and total_warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=start_lr / base_lr,
            total_iters=total_warmup_steps
        )

        total_steps = total_epochs * train_steps_per_epoch
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=float(scheduler_config["eta_min"])
        )

        combined_scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])
        return combined_scheduler

    else:
        total_steps = total_epochs * train_steps_per_epoch
        return CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=float(scheduler_config["eta_min"])
        )


def create_scheduler(optimizer, config: dict, total_steps: int):
    scheduler_type = config['type'].lower()
    if scheduler_type == "cosineannealinglr":
        eta_min = float(config['eta_min'])
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def get_detection_model(num_classes: int, backbone_config_path: str, device: torch.device) -> nn.Module:
    model = YOLOv3(num_classes=num_classes, backbone_config_path=backbone_config_path)
    model.to(device)
    return model


def get_yolo_anchors(anchors_list: dict, device: torch.device) -> List[torch.Tensor]:
    return [torch.tensor(a, dtype=torch.float32, device=device) for a in anchors_list]


def convert_to_zero_indexed_list(indexes: List[int]) -> List[int]:
    return [i - 1 for i in indexes]


def invert_index_list(indexes: List[int]) -> List[int]:
    max_index = max(indexes)
    inverted = [-1] * (max_index + 1)
    for i, idx in enumerate(indexes):
        inverted[idx] = i
    return inverted
