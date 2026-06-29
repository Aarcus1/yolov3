import torch
from torch.amp import GradScaler
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from typing import Optional, Dict, Any, List, Tuple, Callable
from tqdm import tqdm

from source.serialization.serialization import save_checkpoint
from source.utilities.metrics import detection_metrics
from source.utilities.utilities import *
from source.postprocess.nms import *
from source.my_logging.my_logging import *
from source.utilities.data_conversion import *
from source.utilities.subset_type import SplitType
from source.sampling.sampler import *


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scaler: GradScaler,
            scheduler,
            loss_fn: Callable,
            device: torch.device,
            main_config: Dict[str, Any],
            snapshot_path_base: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        if main_config["training"]["gradient_clipping"]["enabled"]:
            self.grad_clip: Optional[float] = main_config["training"]["gradient_clipping"]["max_absolute_value"]

        else:
            self.grad_clip: Optional[float] = None
        self.use_amp: bool = main_config['training']["use_amp"]
        self.scaler = scaler

        anchors_list = main_config['model']['anchors']

        self.anchors: List[torch.Tensor] = [torch.tensor(a, dtype=torch.float32, device=device) for a in anchors_list]
        self.strides: List[int] = main_config['model']['strides']
        self.num_classes: int = main_config['dataset']['class_count']

        self.confidence_threshold = main_config['processing']['nms']['confidence_threshold']
        self.iou_threshold = main_config['processing']['nms']['iou_threshold']
        self.iou_ignore_threshold = main_config['processing']['yolo']['iou_ignore_threshold']

        self.snapshot_path_base = snapshot_path_base
        self.class_names = main_config["dataset"]["class_names"]

        self.packed_index_to_original_index_mapping = convert_to_zero_indexed_list(main_config["dataset"]["ids"])
        self.original_index_to_packed_index_mapping = invert_index_list(self.packed_index_to_original_index_mapping)

        self.sample_directory = main_config["training"]["sample_directory"]

        self.scheduler = scheduler

        self.warmup_enabled = main_config["training"].get("warmup", {}).get("enabled", False)
        self.warmup_epochs = main_config["training"].get("warmup", {}).get("epochs", 0)
        self.start_lr = main_config["training"].get("warmup", {}).get("start_lr", 1e-6)
        self.base_lr = self.optimizer.param_groups[0]['lr']

        self.validation_normalization_mean = main_config["dataset"]["validation"]["augmentation"]["normalize"]["mean"]
        self.validation_normalization_std = main_config["dataset"]["validation"]["augmentation"]["normalize"]["std"]

        image_size = main_config['dataset']['image_size']
        if len(image_size) == 2:
            self.image_size: Tuple[int, int] = (image_size[0], image_size[1])
        else:
            raise ValueError("Image size must have 2 dimensions")

    def _run_epoch(
            self,
            dataloader: torch.utils.data.DataLoader,
            logger: Logger,
            epoch: int,
            split_type: SplitType
    ) -> Dict[str, float]:
        if split_type != SplitType.TRAIN and split_type != SplitType.VALIDATION:
            raise ValueError("split_type must be either TRAIN or VALIDATION")

        if split_type == SplitType.TRAIN:
            self.model.train()
        elif split_type == SplitType.VALIDATION:
            self.model.eval()

        total_loss = 0.0
        num_samples = 0

        all_outputs = []
        all_targets = []

        for step_number, (inputs, targets) in enumerate(dataloader):
            inputs = torch.stack(inputs).to(self.device, non_blocking=True)
            total_step_number = (epoch - 1) * len(dataloader) + step_number + 1

            # gt boxes converted to format compatible with loss calculation
            preprocessed_targets = build_yolo_targets(
                targets,
                image_size=self.image_size,
                anchors=self.anchors,
                strides=self.strides,
                num_classes=self.num_classes,
                device=self.device,
                iou_ignore_thresh=self.iou_ignore_threshold,
                original_index_to_packed_index_mapping=self.original_index_to_packed_index_mapping,
            )

            if split_type == SplitType.TRAIN:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(self.device.type, enabled=self.scaler.is_enabled()):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, preprocessed_targets)

            if split_type == SplitType.TRAIN:
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                else:
                    loss.backward()
                    if self.grad_clip is not None:
                        clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()

            batch_size = inputs.size(0)
            total_loss += loss.item()

            if split_type == SplitType.VALIDATION:
                decoded_batch = decode_batch_outputs(outputs, self.image_size, self.anchors, self.strides)
                postprocessed_outputs = apply_nms(decoded_batch, self.confidence_threshold, self.iou_threshold)
                postprocessed_targets = coco_gts_to_metrics(targets, self.image_size, False,
                                                            self.original_index_to_packed_index_mapping, self.device)

                all_outputs = all_outputs + postprocessed_outputs
                all_targets = all_targets + postprocessed_targets

                # Only save first btach of images
                if step_number == 0:
                    save_sample_predictions(
                        images=inputs,
                        predictions=postprocessed_outputs,
                        class_names=self.class_names,
                        snapshot_path_base=self.snapshot_path_base,
                        sample_directory=self.sample_directory,
                        epoch=epoch,
                        mean=self.validation_normalization_mean,
                        std=self.validation_normalization_std,
                    )

            step_metrics = {"loss": loss.item() / batch_size}
            logger.log_step(step_metrics, total_step_number, split_type)

            num_samples += batch_size

        batch_metrics = {"loss": total_loss / num_samples}

        if split_type == SplitType.VALIDATION:
            batch_metrics_list = detection_metrics(
                all_outputs,
                all_targets
            )

            for _, (metric, value) in enumerate(batch_metrics_list):
                batch_metrics[metric] = value

        logger.log_epoch(batch_metrics, epoch, split_type)

        return batch_metrics

    def train_epoch(self, dataloader: torch.utils.data.DataLoader, logger: Logger, epoch: int):
        return self._run_epoch(dataloader, logger=logger, epoch=epoch, split_type=SplitType.TRAIN)

    @torch.no_grad()
    def validate_epoch(self, dataloader: torch.utils.data.DataLoader, logger, epoch: int):
        return self._run_epoch(dataloader, logger=logger, epoch=epoch, split_type=SplitType.VALIDATION)

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict, strict: bool = True):
        self.model.load_state_dict(state["model"], strict=strict)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.use_amp and state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])

    def train(
            self,
            training_loader: torch.utils.data.DataLoader,
            validation_loader: torch.utils.data.DataLoader,
            logger: Logger,
            main_config: Dict[str, Any],
            max_epoch: int,
            start_epoch: int = 1,
    ) -> None:
        backbone_config = main_config['training']['checkpoints']['backbone']
        validation_interval = main_config['training']['validation_interval_epochs']
        checkpoint_interval = main_config['training']['checkpoints']['save_interval_epochs']

        # Freeze backbone
        if backbone_config["enabled"]:
            self.model.freeze_backbone()

        max_train_step_count = max_epoch * len(training_loader)
        starting_train_step_count = (start_epoch - 1) * len(training_loader)

        logger.init_train_bar(max_train_step_count, starting_train_step_count)

        for current_epoch in range(start_epoch, max_epoch + 1):

            # Unfreeze backbone
            if backbone_config["enabled"] and self.model.is_backbone_frozen() and \
                    backbone_config["freeze_backbone_epoch_count"] < current_epoch:
                self.model.unfreeze_backbone()

            # Training
            self._train_epoch_wrapper(training_loader, logger, current_epoch)

            # Validation
            if current_epoch % validation_interval == 0 or max_epoch == current_epoch:
                self._validate_epoch_wrapper(validation_loader, logger, current_epoch)

            # Checkpoint saving
            if current_epoch % checkpoint_interval == 0 or max_epoch == current_epoch:
                self._save_checkpoint(current_epoch)

    def _train_epoch_wrapper(
            self,
            training_loader: torch.utils.data.DataLoader,
            logger: Logger,
            epoch: int
    ) -> None:
        try:
            self.train_epoch(training_loader, logger, epoch)
        except Exception as e:
            logger.close_train_bar()
            raise RuntimeError(f"Training failed at epoch {epoch}: {e}")

    def _validate_epoch_wrapper(
            self,
            validation_loader: torch.utils.data.DataLoader,
            logger: Logger,
            epoch: int
    ) -> None:
        try:
            logger.init_validation_bar(len(validation_loader))
            self.validate_epoch(validation_loader, logger, epoch)
            logger.close_validation_bar()
        except Exception as e:
            logger.close_validation_bar()
            print(f"Validation failed at epoch {epoch}: {e}")

    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint_path = self.snapshot_path_base + f"_epo{epoch}.pth"
        save_checkpoint(
            state=self.state_dict(),
            checkpoint_path=checkpoint_path,
            epoch=epoch
        )
