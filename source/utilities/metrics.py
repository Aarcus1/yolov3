import torch
from typing import List, Dict, Tuple

from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou
from source.utilities.data_conversion import build_coco_dt, build_coco_gt

NameMetricPair = Tuple[str, float]

def evaluate_coco(outputs, targets):
    coco_gt = build_coco_gt(targets)
    coco_dt = coco_gt.loadRes(build_coco_dt(outputs))

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval

def detection_metrics(
    outputs: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
) -> List[NameMetricPair]:

    coco_eval = evaluate_coco(outputs, targets)

    metrics: List[NameMetricPair] = [
        ("mAP", float(coco_eval.stats[0])),          # AP@[0.50:0.95]
        ("mAP50", float(coco_eval.stats[1])),        # AP@0.50
        ("mAP75", float(coco_eval.stats[2])),        # AP@0.75
        ("mAP_small", float(coco_eval.stats[3])),
        ("mAP_medium", float(coco_eval.stats[4])),
        ("mAP_large", float(coco_eval.stats[5])),
        ("AR@1", float(coco_eval.stats[6])),
        ("AR@10", float(coco_eval.stats[7])),
        ("AR@100", float(coco_eval.stats[8])),
        ("AR_large", float(coco_eval.stats[11])),
    ]

    return metrics