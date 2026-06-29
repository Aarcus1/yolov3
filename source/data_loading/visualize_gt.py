"""
Visualize ground-truth bounding boxes for the first N images of a COCO dataset split.

Usage:
    python source/data_loading/visualize_gt.py \
        --annotation_file data/coco_2017/annotations/instances_training.json \
        --image_dir       data/coco_2017/images/training \
        --output_dir      samples/gt_vis \
        --num_images      20
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torchvision.datasets import CocoDetection

# Allow bare imports when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from source.sampling.sampler import _draw_predictions_on_image, _save_annotated_image


def _pil_to_numpy_rgb(pil_image: object) -> np.ndarray:
    return np.array(pil_image, dtype=np.uint8)


def _coco_xywh_to_xyxy_tensor(annotations: list) -> torch.Tensor:
    if not annotations:
        return torch.zeros((0, 4), dtype=torch.float32)
    boxes = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
    return torch.tensor(boxes, dtype=torch.float32)


def _build_class_name_map(coco_api) -> list:
    categories = sorted(coco_api.loadCats(coco_api.getCatIds()), key=lambda c: c["id"])
    return [cat["name"] for cat in categories]


def _category_id_to_index(coco_api) -> dict:
    categories = sorted(coco_api.loadCats(coco_api.getCatIds()), key=lambda c: c["id"])
    return {cat["id"]: idx for idx, cat in enumerate(categories)}


def visualize_gt(annotation_file: str, image_dir: str, output_dir: str, num_images: int) -> None:
    dataset = CocoDetection(root=image_dir, annFile=annotation_file)
    coco_api = dataset.coco

    class_names = _build_class_name_map(coco_api)
    cat_id_to_idx = _category_id_to_index(coco_api)

    os.makedirs(output_dir, exist_ok=True)

    n = min(num_images, len(dataset))
    print(f"Saving {n} annotated images to '{output_dir}' ...")

    for i in range(n):
        pil_image, annotations = dataset[i]
        image = _pil_to_numpy_rgb(pil_image)

        boxes = _coco_xywh_to_xyxy_tensor(annotations)
        labels = torch.tensor(
            [cat_id_to_idx[ann["category_id"]] for ann in annotations],
            dtype=torch.long,
        )
        scores = torch.ones(len(annotations), dtype=torch.float32)

        annotated = _draw_predictions_on_image(image, boxes, scores, labels, class_names)
        _save_annotated_image(annotated, output_dir, i)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize COCO ground-truth boxes for the first N images."
    )
    parser.add_argument(
        "--annotation_file", required=True,
        help="Path to the COCO annotation JSON (e.g. instances_training.json)",
    )
    parser.add_argument(
        "--image_dir", required=True,
        help="Directory containing the corresponding COCO images",
    )
    parser.add_argument(
        "--output_dir", default="samples/gt_vis",
        help="Directory where annotated images will be saved (default: samples/gt_vis)",
    )
    parser.add_argument(
        "--num_images", type=int, default=20,
        help="Number of images to visualize (default: 20)",
    )
    args = parser.parse_args()

    visualize_gt(
        annotation_file=args.annotation_file,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
    )


if __name__ == "__main__":
    main()

