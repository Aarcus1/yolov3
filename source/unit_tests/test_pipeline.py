import unittest
import os
import torch
from source.data_loading.data_loading import *
import yaml

from source.postprocess.nms import apply_nms
from source.sampling.sampler import save_sample_predictions
from source.utilities.data_conversion import coco_gts_to_metrics, build_yolo_targets, decode_batch_outputs
from source.utilities.utilities import convert_to_zero_indexed_list, invert_index_list

yaml_dataset_config_string = """
image_size: [416, 416]
class_names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
class_count: 80
training:
  image_directory_path: "./source/unit_tests/data/coco_2017/images/training"
  annotation_file_path: "./source/unit_tests/data/coco_2017/annotations/instances_training.json"
  batch_size: 32
  do_shuffle: false
  do_pin_memory: false
  worker_count: 0
  augmentation: {}
validation:
  image_directory_path: "./source/unit_tests/data/coco_2017/images/validation"
  annotation_file_path: "./source/unit_tests/data/coco_2017/annotations/instances_validation.json"
  batch_size: 8
  do_shuffle: false
  do_pin_memory: false
  worker_count: 0
  augmentation: {}
testing:
  image_directory_path: "./source/unit_tests/data/coco_2017/images/validation"
  annotation_file_path: "./source/unit_tests/data/coco_2017/annotations/instances_validation.json"
  batch_size: 1
  do_shuffle: false
  do_pin_memory: false
  worker_count: 0
  augmentation: {}
"""
yaml_dataset_config = yaml.safe_load(yaml_dataset_config_string)

other_config_string = """
anchors:
    - [ [ 10, 13 ], [ 16, 30 ], [ 33, 23 ] ]
    - [ [ 30, 61 ], [ 62, 45 ], [ 59, 119 ] ]
    - [ [ 116, 90 ], [ 156, 198 ], [ 373, 326 ] ]
strides: [ 8, 16, 32 ]
"""
other_config = yaml.safe_load(other_config_string)

def tensors_close(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.allclose(a, b, atol=1e-4)
    return a == b

def sort_detections_inplace(target):
    if len(target["boxes"]) == 0:
        return target

    boxes = target["boxes"]

    # Sort by x1, then y1, then x2, then y2
    sort_values = (
        boxes[:, 0] * 1e9 +
        boxes[:, 1] * 1e6 +
        boxes[:, 2] * 1e3 +
        boxes[:, 3]
    )

    indices = torch.argsort(sort_values)

    target["boxes"] = target["boxes"][indices]
    target["labels"] = target["labels"][indices]

    if "scores" in target:
        target["scores"] = target["scores"][indices]

    return target

def dicts_close(d1, d2):
    for k in d1:
        if k not in d2:
            return False
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, torch.Tensor):
            if not tensors_close(v1, v2):
                return False
        elif isinstance(v1, list):
            if len(v1) != len(v2):
                return False
            for x, y in zip(v1, v2):
                if isinstance(x, torch.Tensor):
                    if not tensors_close(x, y):
                        return False
                else:
                    if x != y:
                        return False
        else:
            if v1 != v2:
                return False
    return True

class TestStringMethods(unittest.TestCase):
    def test_input_output_same(self):
        test_loader = get_coco_data_loader(yaml_dataset_config['testing'],
                                           yaml_dataset_config['image_size'])
        _, gt_targets = next(iter(test_loader))

        packed_index_to_original_index_mapping = convert_to_zero_indexed_list(yaml_dataset_config["ids"])
        original_index_to_packed_index_mapping = invert_index_list(packed_index_to_original_index_mapping)
        device = torch.device("cpu")

        anchors_list = other_config['anchors']
        anchors = [torch.tensor(a, dtype=torch.float32, device=device) for a in anchors_list]
        strides = other_config['strides']
        model_targets = build_yolo_targets(
            gt_targets,
            image_size=yaml_dataset_config['image_size'],
            anchors=anchors,
            strides=strides,
            num_classes=yaml_dataset_config['class_count'],
            device=device,
            iou_ignore_thresh=1.0,
            original_index_to_packed_index_mapping=original_index_to_packed_index_mapping,
        )
        decoded_batch = decode_batch_outputs(model_targets, yaml_dataset_config['image_size'], anchors, strides, False)
        metric_targets_1 = apply_nms(decoded_batch, 0.5, 1.0)
        metric_targets_2 = coco_gts_to_metrics(gt_targets, yaml_dataset_config['image_size'], False,
                                               original_index_to_packed_index_mapping, device)

        metric_targets_1 = [sort_detections_inplace(t) for t in metric_targets_1]
        metric_targets_2 = [sort_detections_inplace(t) for t in metric_targets_2]

        self.assertEqual(len(metric_targets_1), len(metric_targets_2))
        for d1, d2 in zip(metric_targets_1, metric_targets_2):
            for key in ["boxes", "labels"]:
                v1, v2 = d1.get(key), d2.get(key)
                self.assertIsNotNone(v1, f"Missing '{key}' in d1: {d1}")
                self.assertIsNotNone(v2, f"Missing '{key}' in d2: {d2}")
                if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                    self.assertTrue(torch.allclose(v1, v2, atol=1e-4), f"Mismatch in {key}: {v1} vs {v2}")
                else:
                    self.assertEqual(v1, v2, f"Mismatch in {key}: {v1} vs {v2}")

if __name__ == "__main__":
    unittest.main()
