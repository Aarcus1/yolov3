from typing import List, Dict, Tuple
import torch
from torchvision.ops import nms


def nms_per_image(
    boxes_list: List[torch.Tensor],
    obj_scores_list: List[torch.Tensor],
    class_scores_list: List[torch.Tensor],
    label_list: List[torch.Tensor],
    conf_threshold: float,
    iou_threshold: float,
) -> Dict[str, torch.Tensor]:
    if not boxes_list:
        return {"boxes": torch.zeros((0, 4), device="cuda"),
                "scores": torch.zeros((0,), device="cuda"),
                "labels": torch.zeros((0,), dtype=torch.long, device="cuda")}

    boxes = torch.cat(boxes_list, dim=0)
    class_scores = torch.cat(class_scores_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    # class_scores already equals obj_score * max_class_prob (computed in decode_yolo_preds),
    # so use it directly — multiplying by obj_scores again would square the objectness term.
    scores = class_scores

    keep_mask = scores > conf_threshold
    boxes, scores, labels = boxes[keep_mask], scores[keep_mask], labels[keep_mask]

    if boxes.numel() == 0:
        return {"boxes": boxes,
                "scores": scores,
                "labels": torch.zeros((0,), dtype=torch.long, device=boxes.device)}

    keep_boxes, keep_scores, keep_labels = [], [], []

    for cls in labels.unique():
        cls_mask = labels == cls
        cls_boxes, cls_scores = boxes[cls_mask], scores[cls_mask]
        keep_idx = nms(cls_boxes, cls_scores, iou_threshold)
        keep_boxes.append(cls_boxes[keep_idx])
        keep_scores.append(cls_scores[keep_idx])
        keep_labels.append(torch.full((keep_idx.numel(),), cls, device=boxes.device, dtype=torch.long))

    return {
        "boxes": torch.cat(keep_boxes, dim=0),
        "scores": torch.cat(keep_scores, dim=0),
        "labels": torch.cat(keep_labels, dim=0),
    }


def apply_nms(
    decoded_batch: List[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]],
    conf_threshold: float,
    iou_threshold: float,
) -> List[Dict[str, torch.Tensor]]:
    results = []

    for boxes_list, obj_scores_list, class_scores_list, label_list in decoded_batch:
        results.append(
            nms_per_image(
                boxes_list, obj_scores_list, class_scores_list, label_list,
                conf_threshold, iou_threshold
            )
        )
    return results
