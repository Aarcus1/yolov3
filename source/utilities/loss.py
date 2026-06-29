import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class YOLOv3Loss(nn.Module):
    def __init__(self, num_classes: int,
                 lambda_coord: float,
                 lambda_obj: float,
                 lambda_noobj: float,
                 lambda_cls: float):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls

    def forward(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        total_loss = 0.0

        for prediction, target in zip(predictions, targets):
            # prediction: [B, A, H, W, 5 + C] - raw logits
            # target:     [B, A, H, W, 5 + C] - encoded targets
            # 5 = (tx, ty, tw, th, tobj), C = num_classes
            objectness_mask = target[..., 4] == 1
            no_objectness_mask = target[..., 4] == 0
            objectness_mask_float = objectness_mask.float()
            no_objectness_mask_float = no_objectness_mask.float()

            # box-size weighting: give more gradient to small boxes
            # target[..., 2:4] stores log(w/anchor), log(h/anchor)
            box_scale = (2.0 - target[..., 2].exp() * target[..., 3].exp()).clamp(min=0.0)

            # xy predictions are raw logits, sigmoid is necessary
            # multiplying by objectness_mask ensures coordinate loss is only applied on the largest iou anchor
            loss_xy = ((prediction[..., 0:2].sigmoid() - target[..., 0:2]) ** 2
                       * objectness_mask_float.unsqueeze(-1)
                       * box_scale.unsqueeze(-1)).sum()

            # targets are already converted to logit space
            # multiplying by objectness_mask ensures coordinate loss is only applied on the largest iou anchor
            loss_wh = ((prediction[..., 2:4] - target[..., 2:4]) ** 2
                       * objectness_mask_float.unsqueeze(-1)
                       * box_scale.unsqueeze(-1)).sum()

            loss_coordinate = self.lambda_coord * (loss_xy + loss_wh)

            # Combine obj/noobj loss with per-cell lambda weighting
            # Ignored object have 0 weight
            combined_obj_weight = objectness_mask_float * self.lambda_obj + no_objectness_mask_float * self.lambda_noobj
            loss_objectness = F.binary_cross_entropy_with_logits(
                prediction[..., 4],
                target[..., 4],
                weight=combined_obj_weight,
                reduction='sum'
            )

            # weighing by objectness_mask ensures coordinate loss is only applied on the largest iou anchor
            prediction_class = prediction[..., 5:]
            target_class = target[..., 5:]
            loss_class = self.lambda_cls * F.binary_cross_entropy_with_logits(
                prediction_class, target_class, weight=objectness_mask_float.unsqueeze(-1), reduction='sum'
            )

            total_loss += loss_coordinate + loss_objectness + loss_class

        return total_loss
