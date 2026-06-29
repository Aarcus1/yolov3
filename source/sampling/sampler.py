import os
from typing import List, Dict
import torch
import numpy as np
import cv2

def _denormalize_images(
    images: torch.Tensor,
    mean: tuple,
    std: tuple,
) -> np.ndarray:
    images = images.detach().cpu().numpy()

    if images.shape[1] == 3:  # (B, C, H, W)
        # Unnormalize: img = img * std + mean
        mean_arr = np.array(mean).reshape(1, 3, 1, 1)
        std_arr = np.array(std).reshape(1, 3, 1, 1)
        images = images * std_arr + mean_arr
        images = np.clip(images, 0, 1)
        images = (images * 255).astype(np.uint8)
        images = images.transpose(0, 2, 3, 1)  # B, H, W, C
    else:
        # Fallback for non-normalized or grayscale images
        images = np.clip(images, 0, 1) if images.max() <= 1 else images / 255.0
        images = (images * 255).astype(np.uint8)
        images = images.transpose(0, 2, 3, 1)

    return images

def _draw_predictions_on_image(
    image: np.ndarray,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
) -> np.ndarray:
    img_annotated = image.copy()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.int().tolist()
        class_id = int(label)
        text = f"{class_names[class_id]}: {score:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_annotated, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return img_annotated

def _save_annotated_image(
    image: np.ndarray,
    output_dir: str,
    image_index: int,
) -> None:
    filename = os.path.join(output_dir, f"{image_index}.png")
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def save_sample_predictions(
    images: torch.Tensor,
    predictions: List[Dict[str, torch.Tensor]],
    class_names: List[str],
    snapshot_path_base: str,
    sample_directory: str,
    epoch: int,
    mean: tuple,
    std: tuple,
) -> None:
    output_dir = os.path.join(sample_directory, os.path.basename(snapshot_path_base), f"epoch{epoch}")
    os.makedirs(output_dir, exist_ok=True)

    denormalized_images = _denormalize_images(images, mean, std)

    for i, (image, prediction) in enumerate(zip(denormalized_images, predictions)):
        boxes = prediction["boxes"].cpu()
        scores = prediction["scores"].cpu()
        labels = prediction["labels"].cpu()

        annotated_image = _draw_predictions_on_image(image, boxes, scores, labels, class_names)

        _save_annotated_image(annotated_image, output_dir, i)

