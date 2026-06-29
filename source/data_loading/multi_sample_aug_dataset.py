import random
from typing import List, Dict, Tuple
from enum import Enum

import cv2
import numpy as np
from torch.utils.data import Dataset


class AugmentationType(Enum):
    NONE = "none"
    MOSAIC = "mosaic"
    MIXUP = "mixup"


class MultiSampleAugDataset(Dataset):

    def __init__(
        self,
        base_dataset: Dataset,
        image_size: Tuple[int, int],
        mosaic_prob: float,
        mixup_prob: float,
        mixup_alpha: float = 0.5,
        transform_fn=None,
    ):
        self.base_dataset = base_dataset
        self.image_size   = image_size
        self.mosaic_prob  = mosaic_prob
        self.mixup_prob   = mixup_prob
        self.mixup_alpha  = mixup_alpha
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, List[Dict], AugmentationType]:
        if random.random() < self.mosaic_prob:
            image, target = self._mosaic(index)
            aug_type = AugmentationType.MOSAIC
        elif random.random() < self.mixup_prob:
            image, target = self._mixup(index)
            aug_type = AugmentationType.MIXUP
        else:
            image, target = self._load_raw(index)
            aug_type = AugmentationType.NONE

        if self.transform_fn is not None:
            image, target = self.transform_fn(image, target, aug_type)

        return image, target, aug_type


    def _load_raw(self, index: int) -> Tuple[np.ndarray, List[Dict]]:
        image, target = self.base_dataset[index]
        return np.array(image), target

    def _random_index(self) -> int:
        return random.randrange(len(self.base_dataset))


    def _mosaic(self, index: int) -> Tuple[np.ndarray, List[Dict]]:
        """
        Stitch 4 images into a single canvas:

          +----------+----------+
          |  img 0   |  img 1   |   cx, cy sampled in [25%, 75%] of output
          +----cx,cy-+----------+   so every tile gets a meaningful slice.
          |  img 2   |  img 3   |
          +----------+----------+
        """
        out_H, out_W = self.image_size
        cx = int(random.uniform(0.25, 0.75) * out_W)
        cy = int(random.uniform(0.25, 0.75) * out_H)

        indices = [index, self._random_index(), self._random_index(), self._random_index()]
        quadrants = [
            (0,  0,  cx,    cy),
            (cx, 0,  out_W, cy),
            (0,  cy, cx,    out_H),
            (cx, cy, out_W, out_H),
        ]

        canvas = np.zeros((out_H, out_W, 3), dtype=np.uint8)
        merged_target: List[Dict] = []

        for idx, (qx1, qy1, qx2, qy2) in zip(indices, quadrants):
            img, anns = self._load_raw(idx)
            src_h, src_w = img.shape[:2]
            q_w, q_h = qx2 - qx1, qy2 - qy1

            canvas[qy1:qy2, qx1:qx2] = cv2.resize(img, (q_w, q_h), interpolation=cv2.INTER_LINEAR)

            scale_x = q_w / src_w
            scale_y = q_h / src_h

            for ann in anns:
                x, y, w, h = ann["bbox"]
                x_new = x * scale_x + qx1
                y_new = y * scale_y + qy1
                w_new = w * scale_x
                h_new = h * scale_y

                x_clip  = max(x_new, qx1)
                y_clip  = max(y_new, qy1)
                x2_clip = min(x_new + w_new, qx2)
                y2_clip = min(y_new + h_new, qy2)
                cw = x2_clip - x_clip
                ch = y2_clip - y_clip

                if cw <= 1 or ch <= 1:
                    continue
                if w_new * h_new > 0 and (cw * ch) / (w_new * h_new) < 0.3:
                    continue

                merged_target.append({**ann, "bbox": [x_clip, y_clip, cw, ch]})

        return canvas, merged_target

    def _mixup(self, index: int) -> Tuple[np.ndarray, List[Dict]]:
        """
        Blend two images pixel-wise:
            out = lam * img_A + (1 - lam) * img_B
            lam ~ Beta(alpha, alpha)

        Boxes from both images are resized to the output size and concatenated.
        No box clipping is needed - both images are simply resized to image_size.
        """
        out_H, out_W = self.image_size

        img_a, anns_a = self._load_raw(index)
        img_b, anns_b = self._load_raw(self._random_index())

        src_h_a, src_w_a = img_a.shape[:2]
        src_h_b, src_w_b = img_b.shape[:2]

        img_a = cv2.resize(img_a, (out_W, out_H), interpolation=cv2.INTER_LINEAR)
        img_b = cv2.resize(img_b, (out_W, out_H), interpolation=cv2.INTER_LINEAR)

        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))

        blended = (lam * img_a.astype(np.float32) +
                   (1.0 - lam) * img_b.astype(np.float32)).clip(0, 255).astype(np.uint8)

        def scale_anns(anns, src_w, src_h):
            scale_x = out_W / src_w
            scale_y = out_H / src_h
            return [{**ann, "bbox": [
                ann["bbox"][0] * scale_x,
                ann["bbox"][1] * scale_y,
                ann["bbox"][2] * scale_x,
                ann["bbox"][3] * scale_y,
            ]} for ann in anns]

        merged_target = scale_anns(anns_a, src_w_a, src_h_a) + scale_anns(anns_b, src_w_b, src_h_b)

        return blended, merged_target

