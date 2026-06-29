from typing import Optional, Dict, Any, List, Tuple, Callable
from typing import Dict, List
import torch
from pycocotools.coco import COCO


def _calculate_ious(wh1: torch.Tensor, wh2: torch.Tensor, eps=1e-9):
    inter = torch.min(wh1[:, None, :], wh2[None, :, :]).prod(-1)
    union = wh1.prod(-1)[:, None] + wh2.prod(-1)[None, :] - inter
    return inter / (union + eps)

def _extract_annotation_data(
    annotations: List[Dict],
    original_index_to_packed_index_mapping: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    box_dimensions = torch.tensor([annotation["bbox"][2:] for annotation in annotations], device=device)
    centers = torch.tensor([
        [(annotation["bbox"][0] + annotation["bbox"][2]/2.0),
         (annotation["bbox"][1] + annotation["bbox"][3]/2.0)]
        for annotation in annotations
    ], device=device)
    classes = torch.tensor(
        [original_index_to_packed_index_mapping[int(annotation["category_id"] - 1)] for annotation in annotations],
        device=device
    )
    return box_dimensions, centers, classes


def _assign_best_anchors(
    box_dimensions: torch.Tensor,
    anchors_on_device: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    best_iou = torch.zeros(len(box_dimensions), device=device)
    best_scale = torch.zeros(len(box_dimensions), dtype=torch.long, device=device)
    best_anchor = torch.zeros(len(box_dimensions), dtype=torch.long, device=device)

    for scale_index, anchor_set in enumerate(anchors_on_device):
        ious = _calculate_ious(box_dimensions, anchor_set)
        max_ious, max_idx = ious.max(dim=1)
        update_mask = max_ious > best_iou
        best_iou[update_mask] = max_ious[update_mask]
        best_scale[update_mask] = scale_index
        best_anchor[update_mask] = max_idx[update_mask]

    return best_iou, best_scale, best_anchor


def _mark_ignored_anchors(
    yolo_targets: List[torch.Tensor],
    batch_index: int,
    box_dimensions: torch.Tensor,
    box_centers: torch.Tensor,
    anchors_on_device: List[torch.Tensor],
    strides: List[int],
    iou_ignore_thresh: float
) -> None:
    for scale_index, anchor_set in enumerate(anchors_on_device):
        stride = strides[scale_index]
        t = yolo_targets[scale_index]

        for i in range(len(box_dimensions)):
            ious = _calculate_ious(box_dimensions[i].unsqueeze(0), anchor_set).squeeze(0)
            ignore_mask = ious > iou_ignore_thresh

            grid_col = int(box_centers[i, 0] / stride)
            grid_row = int(box_centers[i, 1] / stride)
            t[batch_index, ignore_mask, grid_row, grid_col, 4] = -1.0


def _mark_positive_targets(
    yolo_targets: List[torch.Tensor],
    batch_index: int,
    box_dimensions: torch.Tensor,
    box_centers: torch.Tensor,
    class_ids: torch.Tensor,
    best_scale: torch.Tensor,
    best_anchor: torch.Tensor,
    anchors_on_device: List[torch.Tensor],
    strides: List[int],
) -> None:
    for i in range(len(box_dimensions)):
        scale_index = best_scale[i].item()
        anchor_index = best_anchor[i].item()
        stride = strides[scale_index]

        grid_col = int(box_centers[i, 0] / stride)
        grid_row = int(box_centers[i, 1] / stride)
        tx = box_centers[i, 0] / stride - grid_col
        ty = box_centers[i, 1] / stride - grid_row

        if not (0 <= tx <= 1) or not (0 <= ty <= 1):
            raise ValueError("tx or ty out of bounds")

        t = yolo_targets[scale_index]
        w, h = box_dimensions[i]
        anchor_w, anchor_h = anchors_on_device[scale_index][anchor_index]

        t[batch_index, anchor_index, grid_row, grid_col, 0] = tx
        t[batch_index, anchor_index, grid_row, grid_col, 1] = ty
        t[batch_index, anchor_index, grid_row, grid_col, 2] = torch.log(w / anchor_w + 1e-9)
        t[batch_index, anchor_index, grid_row, grid_col, 3] = torch.log(h / anchor_h + 1e-9)
        t[batch_index, anchor_index, grid_row, grid_col, 4] = 1.0
        t[batch_index, anchor_index, grid_row, grid_col, 5 + class_ids[i]] = 1.0


def build_yolo_targets(
    gt_targets: List[List[Dict]],
    image_size: Tuple[int, int],
    anchors: List[torch.Tensor],
    strides: List[int],
    num_classes: int,
    device: torch.device,
    iou_ignore_thresh: float,
    original_index_to_packed_index_mapping: List[int],
) -> List[torch.Tensor]:

    batch_size = len(gt_targets)
    img_h, img_w = image_size
    yolo_targets = []

    for scale_idx, stride in enumerate(strides):
        A = anchors[scale_idx].size(0)
        H, W = img_h // stride, img_w // stride
        t = torch.zeros(batch_size, A, H, W, 5 + num_classes, device=device)
        yolo_targets.append(t)

    anchors_on_device = [a.to(device) for a in anchors]

    for batch_index, annotations in enumerate(gt_targets):
        if len(annotations) == 0:
            continue

        box_dimensions, box_centers, box_classes = _extract_annotation_data(
            annotations, original_index_to_packed_index_mapping, device
        )

        _mark_ignored_anchors(
            yolo_targets, batch_index, box_dimensions, box_centers, anchors_on_device, strides,
            iou_ignore_thresh
        )

        best_iou, best_scale, best_anchor = _assign_best_anchors(
            box_dimensions, anchors_on_device, device
        )

        _mark_positive_targets(
            yolo_targets, batch_index, box_dimensions, box_centers, box_classes, best_scale, best_anchor,
            anchors_on_device, strides
        )

    return yolo_targets


def flatten_annotations(
        gts: List[List[Dict[str, Any]]],
        original_index_to_packed_index_mapping: List[int],
        device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    all_boxes, all_labels, all_iscrowd = [], [], []
    num_anns_per_image = []

    for image_anns in gts:
        num_anns_per_image.append(len(image_anns))
        for ann in image_anns:
            x, y, w, h = ann["bbox"]
            all_boxes.append([x, y, w, h])
            all_labels.append(original_index_to_packed_index_mapping[int(ann["category_id"]) - 1])
            all_iscrowd.append(ann.get("iscrowd", 0))

    if all_boxes:
        all_boxes = torch.tensor(all_boxes, dtype=torch.float32, device=device)
        all_labels = torch.tensor(all_labels, dtype=torch.int64, device=device)
        all_iscrowd = torch.tensor(all_iscrowd, dtype=torch.int64, device=device)
    else:
        all_boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
        all_labels = torch.zeros((0,), dtype=torch.int64, device=device)
        all_iscrowd = torch.zeros((0,), dtype=torch.int64, device=device)

    return all_boxes, all_labels, all_iscrowd, num_anns_per_image


def convert_boxes_xywh_to_xyxy(boxes: torch.Tensor, image_size: Tuple[int, int], normalize: bool) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    if normalize:
        img_h, img_w = image_size
        scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=boxes.device)
        boxes /= scale

    return boxes


def split_annotations_per_image(
        boxes: torch.Tensor,
        labels: torch.Tensor,
        iscrowd: torch.Tensor,
        num_anns_per_image: List[int]
) -> List[Dict[str, torch.Tensor]]:
    results = []
    start = 0
    for n in num_anns_per_image:
        if n == 0:
            results.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=boxes.device),
                "labels": torch.zeros((0,), dtype=torch.int64, device=boxes.device),
                "iscrowd": torch.zeros((0,), dtype=torch.int64, device=boxes.device),
            })
        else:
            results.append({
                "boxes": boxes[start:start + n],
                "labels": labels[start:start + n],
                "iscrowd": iscrowd[start:start + n],
            })
        start += n
    return results


def coco_gts_to_metrics(
        gts: List[List[Dict[str, Any]]],
        image_size: Tuple[int, int],
        normalize: bool,
        original_index_to_packed_index_mapping: List[int],
        device: torch.device
) -> List[Dict[str, torch.Tensor]]:
    all_boxes, all_labels, all_iscrowd, num_anns_per_image = flatten_annotations(
        gts, original_index_to_packed_index_mapping, device
    )

    all_boxes = convert_boxes_xywh_to_xyxy(all_boxes, image_size, normalize)

    results = split_annotations_per_image(all_boxes, all_labels, all_iscrowd, num_anns_per_image)

    return results

def decode_yolo_preds(
    preds: List[torch.Tensor],
    image_size: Tuple[int, int],
    anchors: List[torch.Tensor],
    strides: List[int],
    are_logits: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    device = preds[0].device
    boxes_list, obj_scores_list, class_scores_list, labels_list = [], [], [], []

    for p, anchor, stride in zip(preds, anchors, strides):
        A, H, W, _ = p.shape
        C = p.shape[-1] - 5

        # indexing="xy": grid_x shape [H, W] with col offsets, grid_y shape [H, W] with row offsets
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            indexing="xy",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]: [...,0]=x col, [...,1]=y row

        tx, ty, tw, th = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        tobj, tcls = p[..., 4], p[..., 5:]

        if are_logits:
            cx = (tx.sigmoid() + grid[..., 0]) * stride
            cy = (ty.sigmoid() + grid[..., 1]) * stride
            obj_scores = tobj.sigmoid().view(-1)
            class_probs = tcls.sigmoid().reshape(-1, C)
            final_scores = class_probs * obj_scores.unsqueeze(1)
            class_scores, labels = final_scores.max(dim=1)
        else:
            cx = (tx + grid[..., 0]) * stride
            cy = (ty + grid[..., 1]) * stride
            obj_scores = tobj.view(-1)
            class_probs = tcls.reshape(-1, C)
            final_scores = class_probs * obj_scores.unsqueeze(1)
            class_scores, labels = final_scores.max(dim=1)

        anchor_w = anchor[:, 0].view(A, 1, 1)
        anchor_h = anchor[:, 1].view(A, 1, 1)
        w = torch.exp(tw) * anchor_w
        h = torch.exp(th) * anchor_h

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)
        img_h, img_w = image_size
        boxes[:, 0::2].clamp_(0, img_w)
        boxes[:, 1::2].clamp_(0, img_h)

        boxes_list.append(boxes)
        obj_scores_list.append(obj_scores)
        class_scores_list.append(class_scores)
        labels_list.append(labels)

    return boxes_list, obj_scores_list, class_scores_list, labels_list


def decode_batch_outputs(
    outputs: List[torch.Tensor],
    image_size: Tuple[int, int],
    anchors: List[torch.Tensor],
    strides: List[int],
    are_logits: bool = True,
) -> List[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]:
    batch_size = outputs[0].shape[0]
    decoded_results = []

    for b in range(batch_size):
        preds_per_image = [out[b] for out in outputs]
        boxes_list, obj_scores_list, class_scores_list, labels_list = decode_yolo_preds(
            preds_per_image, image_size, anchors, strides, are_logits
        )
        decoded_results.append((boxes_list, obj_scores_list, class_scores_list, labels_list))

    return decoded_results


def build_coco_gt(targets: List[Dict[str, torch.Tensor]]) -> COCO:
    dataset = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    category_ids = set()
    annotation_id = 1

    for image_id, target in enumerate(targets):
        dataset["images"].append(
            {
                "id": image_id,
            }
        )

        boxes = target["boxes"].cpu()
        labels = target["labels"].cpu()
        iscrowd = target.get(
            "iscrowd",
            torch.zeros(len(boxes), dtype=torch.int64)
        ).cpu()

        for box, label, crowd in zip(boxes, labels, iscrowd):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1

            dataset["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, w, h],
                    "area": float(w * h),
                    "iscrowd": int(crowd),
                }
            )

            category_ids.add(int(label))
            annotation_id += 1

    dataset["categories"] = [
        {"id": cid}
        for cid in sorted(category_ids)
    ]

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()

    return coco

def build_coco_dt(outputs: List[Dict[str, torch.Tensor]]) -> List[Dict]:
    detections = []

    for image_id, output in enumerate(outputs):
        boxes = output["boxes"].cpu()
        labels = output["labels"].cpu()
        scores = output["scores"].cpu()

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1

            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, w, h],
                    "score": float(score),
                }
            )

    return detections