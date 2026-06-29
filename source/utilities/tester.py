import torch
from torch.amp import autocast
from typing import Dict, Any, List, Tuple, Callable, Optional

from source.postprocess.nms import apply_nms
from source.utilities.metrics import detection_metrics, NameMetricPair
from source.utilities.utilities import *
from source.my_logging.my_logging import Logger
from source.utilities.data_conversion import *
from source.utilities.subset_type import SplitType
from source.sampling.sampler import save_sample_predictions


class Tester:
    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device,
            main_config: Dict[str, Any],
            snapshot_path_base: str,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        anchors_list = main_config['model']['anchors']
        self.anchors: List[torch.Tensor] = [
            torch.tensor(a, dtype=torch.float32, device=device) for a in anchors_list
        ]
        self.strides: List[int] = main_config['model']['strides']
        self.num_classes: int = main_config['dataset']['class_count']

        self.confidence_threshold = main_config['processing']['nms']['confidence_threshold']
        self.iou_threshold = main_config['processing']['nms']['iou_threshold']
        self.iou_ignore_threshold = main_config['processing']['yolo']['iou_ignore_threshold']

        self.class_names = main_config["dataset"]["class_names"]
        self.snapshot_path_base = snapshot_path_base
        self.sample_directory = os.path.join(main_config["testing"]["save_directory"], "samples")

        self.packed_index_to_original_index_mapping = convert_to_zero_indexed_list(
            main_config["dataset"]["ids"]
        )
        self.original_index_to_packed_index_mapping = invert_index_list(
            self.packed_index_to_original_index_mapping
        )

        image_size = main_config['dataset']['image_size']
        if len(image_size) != 2:
            raise ValueError("Image size must have 2 dimensions")
        self.image_size: Tuple[int, int] = tuple(image_size)

        self.validation_normalization_mean = main_config["dataset"]["testing"]["augmentation"]["normalize"]["mean"]
        self.validation_normalization_std = main_config["dataset"]["testing"]["augmentation"]["normalize"]["std"]

    @torch.no_grad()
    def run(
            self,
            dataloader: torch.utils.data.DataLoader,
            logger: Logger,
    ) -> List[NameMetricPair]:

        num_samples = 0
        all_outputs = []
        all_targets = []

        for step_number, (inputs, targets) in enumerate(dataloader):
            inputs = torch.stack(inputs).to(self.device, non_blocking=True)
            batch_size = inputs.size(0)

            outputs = self.model(inputs)

            decoded_batch = decode_batch_outputs(outputs, self.image_size, self.anchors, self.strides)
            postprocessed_outputs = apply_nms(decoded_batch, self.confidence_threshold, self.iou_threshold)
            all_outputs = all_outputs + postprocessed_outputs

            postprocessed_targets = coco_gts_to_metrics(
                targets,
                self.image_size,
                False,
                self.original_index_to_packed_index_mapping,
                self.device,
            )
            all_targets = all_targets + postprocessed_targets

            if step_number < 100:
                save_sample_predictions(
                    images=inputs,
                    predictions=postprocessed_outputs,
                    class_names=self.class_names,
                    snapshot_path_base=self.snapshot_path_base,
                    sample_directory=self.sample_directory,
                    epoch=step_number,
                    mean=self.validation_normalization_mean,
                    std=self.validation_normalization_std,
                )

            logger.log_step({}, step_number + 1, SplitType.TEST)

            num_samples += batch_size

        batch_metrics_list = detection_metrics(
            all_outputs,
            all_targets
        )

        return batch_metrics_list
