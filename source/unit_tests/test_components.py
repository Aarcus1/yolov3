"""
Unit tests for individual pipeline components:
  - NMS (nms_per_image / apply_nms)
  - calculate_ious
  - LR schedulers (create_combined_scheduler / create_scheduler)
  - invert_index_list / convert_to_zero_indexed_list
  - YOLOv3 model output shapes and backbone freeze/unfreeze
  - detection_metrics / detection_confusion

Loss function tests have been moved to test_loss.py.
"""

import unittest
import math
import torch
import torch.nn as nn

from source.postprocess.nms import nms_per_image, apply_nms
from source.utilities.data_conversion import _calculate_ious
from source.utilities.utilities import (
    create_combined_scheduler,
    create_scheduler,
    invert_index_list,
    convert_to_zero_indexed_list,
)
from source.utilities.metrics import detection_confusion, calculate_mAP50

def _dummy_linear_model(lr=1e-3):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, optimizer


class TestNmsPerImage(unittest.TestCase):
    def _make_inputs(self, boxes, scores, labels):
        """Wrap single lists into the per-scale list format nms_per_image expects."""
        return ([torch.tensor(boxes, dtype=torch.float32)],
                [torch.tensor([math.sqrt(s) for s in scores], dtype=torch.float32)],  # obj scores
                [torch.tensor([math.sqrt(s) for s in scores], dtype=torch.float32)],  # class scores
                [torch.tensor(labels, dtype=torch.long)])

    def test_below_conf_threshold_filtered(self):
        boxes = [[0., 0., 10., 10.]]
        result = nms_per_image(
            [torch.tensor(boxes)],
            [torch.tensor([0.05])],   # obj score
            [torch.tensor([0.05])],   # class score -> combined = 0.0025
            [torch.tensor([0], dtype=torch.long)],
            conf_threshold=0.5,
            iou_threshold=0.5,
        )
        self.assertEqual(result["boxes"].shape[0], 0, "Box below conf threshold should be removed")

    def test_above_conf_threshold_kept(self):
        result = nms_per_image(
            [torch.tensor([[0., 0., 10., 10.]])],
            [torch.tensor([0.9])],
            [torch.tensor([0.9])],
            [torch.tensor([0], dtype=torch.long)],
            conf_threshold=0.5,
            iou_threshold=0.5,
        )
        self.assertEqual(result["boxes"].shape[0], 1)

    def test_duplicate_same_class_suppressed(self):
        """Two heavily overlapping boxes of the same class -> only one survives."""
        boxes = torch.tensor([[10., 10., 50., 50.],
                              [11., 11., 51., 51.]])  # IoU ≈ 0.96
        result = nms_per_image(
            [boxes],
            [torch.tensor([0.9, 0.8])],
            [torch.tensor([0.9, 0.8])],
            [torch.tensor([0, 0], dtype=torch.long)],
            conf_threshold=0.3,
            iou_threshold=0.5,
        )
        self.assertEqual(result["boxes"].shape[0], 1,
                         "Heavily overlapping same-class boxes should collapse to one")

    def test_different_classes_not_suppressed(self):
        """Two overlapping boxes with different classes must both survive."""
        boxes = torch.tensor([[10., 10., 50., 50.],
                              [11., 11., 51., 51.]])
        result = nms_per_image(
            [boxes],
            [torch.tensor([0.9, 0.9])],
            [torch.tensor([0.9, 0.9])],
            [torch.tensor([0, 1], dtype=torch.long)],  # different classes
            conf_threshold=0.3,
            iou_threshold=0.5,
        )
        self.assertEqual(result["boxes"].shape[0], 2,
                         "Different-class boxes must not suppress each other")

    def test_empty_input_returns_empty(self):
        result = nms_per_image([], [], [], [], conf_threshold=0.5, iou_threshold=0.5)
        self.assertEqual(result["boxes"].shape[0], 0)

    def test_all_below_threshold_returns_empty(self):
        result = nms_per_image(
            [torch.tensor([[0., 0., 10., 10.]])],
            [torch.tensor([0.1])],
            [torch.tensor([0.1])],
            [torch.tensor([0], dtype=torch.long)],
            conf_threshold=0.9,
            iou_threshold=0.5,
        )
        self.assertEqual(result["boxes"].shape[0], 0)

    def test_output_keys_present(self):
        result = nms_per_image(
            [torch.tensor([[0., 0., 10., 10.]])],
            [torch.tensor([0.9])],
            [torch.tensor([0.9])],
            [torch.tensor([0], dtype=torch.long)],
            conf_threshold=0.5,
            iou_threshold=0.5,
        )
        for key in ("boxes", "scores", "labels"):
            self.assertIn(key, result, f"Key '{key}' missing from NMS output")

    def test_best_scoring_box_survives_suppression(self):
        """The box with the highest score should be the one that survives."""
        boxes = torch.tensor([[10., 10., 50., 50.],
                              [11., 11., 51., 51.]])
        result = nms_per_image(
            [boxes],
            [torch.tensor([0.6, 0.9])],   # second box has higher score
            [torch.tensor([0.6, 0.9])],
            [torch.tensor([0, 0], dtype=torch.long)],
            conf_threshold=0.3,
            iou_threshold=0.5,
        )
        self.assertEqual(result["boxes"].shape[0], 1)
        self.assertAlmostEqual(result["scores"][0].item(), 0.9 * 0.9, places=3)

    def test_non_overlapping_boxes_all_kept(self):
        """Boxes that don't overlap should all survive regardless of count."""
        boxes = torch.tensor([
            [0.,   0.,  10.,  10.],
            [20.,  0.,  30.,  10.],
            [40.,  0.,  50.,  10.],
        ])
        result = nms_per_image(
            [boxes],
            [torch.tensor([0.9, 0.9, 0.9])],
            [torch.tensor([0.9, 0.9, 0.9])],
            [torch.tensor([0, 0, 0], dtype=torch.long)],
            conf_threshold=0.5,
            iou_threshold=0.5,
        )
        self.assertEqual(result["boxes"].shape[0], 3)


class TestCalculateIous(unittest.TestCase):
    def test_identical_boxes_iou_one(self):
        wh = torch.tensor([[30.0, 40.0]])
        iou = _calculate_ious(wh, wh)
        self.assertAlmostEqual(iou[0, 0].item(), 1.0, places=5)

    def test_non_overlapping_iou_zero(self):
        """Two boxes that share no area when centred at origin have no overlap."""
        wh1 = torch.tensor([[10.0, 10.0]])
        wh2 = torch.tensor([[0.0, 10.0]])   # zero width → no area
        iou = _calculate_ious(wh1, wh2)
        self.assertAlmostEqual(iou[0, 0].item(), 0.0, places=5)

    def test_half_overlap(self):
        """Box1 = 2×2, Box2 = 2×2, intersection = 1×2"""
        wh1 = torch.tensor([[2.0, 2.0]])
        wh2 = torch.tensor([[1.0, 2.0]])
        iou = _calculate_ious(wh1, wh2)
        expected = 0.5
        self.assertAlmostEqual(iou[0, 0].item(), expected, places=5)

    def test_output_shape(self):
        wh1 = torch.rand(5, 2)
        wh2 = torch.rand(3, 2)
        iou = _calculate_ious(wh1, wh2)
        self.assertEqual(iou.shape, (5, 3))

    def test_iou_bounded_zero_to_one(self):
        wh1 = torch.rand(10, 2) * 100
        wh2 = torch.rand(7, 2) * 100
        iou = _calculate_ious(wh1, wh2)
        self.assertTrue((iou >= 0).all())
        self.assertTrue((iou <= 1 + 1e-5).all())

    def test_best_anchor_selected(self):
        """The anchor most similar in size to the box should have the highest IoU."""
        box = torch.tensor([[30.0, 30.0]])
        anchors = torch.tensor([[10.0, 10.0], [30.0, 30.0], [60.0, 60.0]])
        iou = _calculate_ious(box, anchors)   # shape [1, 3]
        best = iou[0].argmax().item()
        self.assertEqual(best, 1, "Anchor 1 (30×30) should have the highest IoU with a 30×30 box")

class TestSchedulers(unittest.TestCase):

    def test_create_scheduler_lr_decreases_over_time(self):
        _, optimizer = _dummy_linear_model(lr=1e-3)
        scheduler = create_scheduler(optimizer, {"type": "cosineannealinglr", "eta_min": 1e-6}, total_steps=100)
        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        self.assertGreater(lrs[0], lrs[-1], "LR should decrease over training")

    def test_create_scheduler_ends_at_eta_min(self):
        eta_min = 1e-6
        _, optimizer = _dummy_linear_model(lr=1e-3)
        scheduler = create_scheduler(optimizer, {"type": "cosineannealinglr", "eta_min": eta_min},
                                     total_steps=200)
        for _ in range(200):
            scheduler.step()
        final_lr = optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(final_lr, eta_min, places=7)

    def test_create_scheduler_t_max_covers_full_training(self):
        """LR should still be above eta_min at the halfway point."""
        eta_min = 1e-6
        base_lr = 1e-3
        _, optimizer = _dummy_linear_model(lr=base_lr)
        total_steps = 1000
        scheduler = create_scheduler(optimizer, {"type": "cosineannealinglr", "eta_min": eta_min},
                                     total_steps=total_steps)
        for _ in range(total_steps // 2):
            scheduler.step()
        mid_lr = optimizer.param_groups[0]['lr']
        self.assertGreater(mid_lr, eta_min,
                           "LR should be above eta_min at the halfway point")
        self.assertLess(mid_lr, base_lr,
                        "LR should be below base_lr at the halfway point")

    def test_combined_scheduler_warmup_increases_lr(self):
        """During warmup, each step should bring the LR closer to base_lr."""
        base_lr = 1e-3
        start_lr = 1e-6
        warmup_epochs = 2
        steps_per_epoch = 10
        _, optimizer = _dummy_linear_model(lr=base_lr)

        scheduler = create_combined_scheduler(
            optimizer,
            scheduler_config={"type": "cosineannealinglr", "eta_min": 1e-7},
            warmup_config={"enabled": True, "epochs": warmup_epochs, "start_lr": start_lr},
            train_steps_per_epoch=steps_per_epoch,
            total_epochs=20,
        )

        warmup_lrs = []
        for _ in range(warmup_epochs * steps_per_epoch):
            warmup_lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # LR must be monotonically increasing during warmup
        for i in range(len(warmup_lrs) - 1):
            self.assertLessEqual(warmup_lrs[i], warmup_lrs[i + 1] + 1e-9,
                                 f"LR decreased during warmup at step {i}")

    def test_combined_scheduler_warmup_reaches_base_lr(self):
        """After warmup steps, LR should be close to base_lr (within 10%)."""
        base_lr = 1e-3
        warmup_epochs = 3
        steps_per_epoch = 10
        _, optimizer = _dummy_linear_model(lr=base_lr)
        scheduler = create_combined_scheduler(
            optimizer,
            scheduler_config={"type": "cosineannealinglr", "eta_min": 1e-7},
            warmup_config={"enabled": True, "epochs": warmup_epochs, "start_lr": 1e-6},
            train_steps_per_epoch=steps_per_epoch,
            total_epochs=20,
        )
        for _ in range(warmup_epochs * steps_per_epoch):
            scheduler.step()
        lr_after_warmup = optimizer.param_groups[0]['lr']
        self.assertGreater(lr_after_warmup, base_lr * 0.9,
                           msg=f"LR should be close to base_lr after warmup, got {lr_after_warmup:.6f}")

    def test_combined_scheduler_no_warmup_falls_back_to_cosine(self):
        """With warmup disabled, LR should behave like plain cosine annealing."""
        base_lr = 1e-3
        _, opt_combined = _dummy_linear_model(lr=base_lr)
        _, opt_plain = _dummy_linear_model(lr=base_lr)

        sched_combined = create_combined_scheduler(
            opt_combined,
            scheduler_config={"type": "cosineannealinglr", "eta_min": 1e-6},
            warmup_config={"enabled": False, "epochs": 0, "start_lr": 1e-6},
            train_steps_per_epoch=10,
            total_epochs=10,
        )
        sched_plain = create_scheduler(opt_plain,
                                       {"type": "cosineannealinglr", "eta_min": 1e-6},
                                       total_steps=100)

        for _ in range(50):
            sched_combined.step()
            sched_plain.step()

        self.assertAlmostEqual(opt_combined.param_groups[0]['lr'],
                               opt_plain.param_groups[0]['lr'], places=6)

class TestIndexMapping(unittest.TestCase):

    def test_convert_to_zero_indexed(self):
        self.assertEqual(convert_to_zero_indexed_list([1, 2, 3]), [0, 1, 2])

    def test_convert_handles_non_contiguous(self):
        self.assertEqual(convert_to_zero_indexed_list([1, 3, 5]), [0, 2, 4])

    def test_invert_round_trip(self):
        """invert_index_list(convert_to_zero_indexed_list(ids))[id-1] == packed_index."""
        ids = [1, 2, 3, 5, 6]   # id 4 is skipped
        zero_indexed = convert_to_zero_indexed_list(ids)
        mapping = invert_index_list(zero_indexed)
        for packed_idx, original_id in enumerate(ids):
            self.assertEqual(mapping[original_id - 1], packed_idx,
                             f"id {original_id} should map to packed index {packed_idx}")

    def test_invert_skipped_ids_are_minus_one(self):
        ids = [1, 3]   # id 2 skipped
        zero_indexed = convert_to_zero_indexed_list(ids)
        mapping = invert_index_list(zero_indexed)
        self.assertEqual(mapping[1], -1, "Skipped id=2 (index 1) should map to -1")

    def test_coco_ids_full_mapping(self):
        """All 80 COCO category ids must map to unique packed indices 0..79."""
        ids = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,
               22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,
               43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,
               62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,
               85,86,87,88,89,90]
        zero_indexed = convert_to_zero_indexed_list(ids)
        mapping = invert_index_list(zero_indexed)
        packed_indices = [mapping[i - 1] for i in ids]
        self.assertEqual(sorted(packed_indices), list(range(80)),
                         "All 80 COCO ids must map to unique packed indices 0..79")

class TestModelOutputShapes(unittest.TestCase):

    def setUp(self):
        from source.model.yolo import YOLOv3
        self.cfg_path = "./configurations/darknet.cfg"
        self.num_classes = 80
        self.model = YOLOv3(num_classes=self.num_classes, backbone_config_path=self.cfg_path)
        self.model.eval()

    def test_output_count(self):
        x = torch.zeros(1, 3, 416, 416)
        outputs = self.model(x)
        self.assertEqual(len(outputs), 3, "YOLOv3 should produce 3 scale outputs")

    def test_output_shapes_416(self):
        B, C = 2, self.num_classes
        x = torch.zeros(B, 3, 416, 416)
        outputs = self.model(x)
        expected_grids = [(52, 52), (26, 26), (13, 13)]  # stride 8, 16, 32
        for i, (out, (H, W)) in enumerate(zip(outputs, expected_grids)):
            self.assertEqual(out.shape, (B, 3, H, W, 5 + C),
                             f"Scale {i}: expected {(B, 3, H, W, 5+C)}, got {tuple(out.shape)}")

    def test_output_last_dim_is_5_plus_classes(self):
        x = torch.zeros(1, 3, 416, 416)
        outputs = self.model(x)
        for i, out in enumerate(outputs):
            self.assertEqual(out.shape[-1], 5 + self.num_classes,
                             f"Scale {i}: last dim should be {5 + self.num_classes}")

    def test_small_batch_size_one(self):
        x = torch.zeros(1, 3, 416, 416)
        outputs = self.model(x)
        for out in outputs:
            self.assertEqual(out.shape[0], 1)

    def test_freeze_backbone_disables_gradients(self):
        self.model.freeze_backbone()
        self.assertTrue(self.model.is_backbone_frozen())
        for param in self.model.backbone.parameters():
            self.assertFalse(param.requires_grad,
                             "Backbone params should have requires_grad=False when frozen")
        # Detection head params must still be trainable
        for param in self.model.head_large.parameters():
            self.assertTrue(param.requires_grad)

    def test_unfreeze_backbone_restores_gradients(self):
        self.model.freeze_backbone()
        self.model.unfreeze_backbone()
        self.assertFalse(self.model.is_backbone_frozen())
        for param in self.model.backbone.parameters():
            self.assertTrue(param.requires_grad,
                            "Backbone params should have requires_grad=True after unfreeze")

    def test_model_forward_no_nan(self):
        torch.manual_seed(0)
        x = torch.randn(1, 3, 416, 416)
        outputs = self.model(x)
        for i, out in enumerate(outputs):
            self.assertFalse(torch.isnan(out).any(), f"NaN in scale {i} output")
            self.assertFalse(torch.isinf(out).any(), f"Inf in scale {i} output")

class TestDetectionMetrics(unittest.TestCase):

    def _make_pred(self, boxes, labels, scores=None):
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        if scores is None:
            scores = [1.0] * len(boxes)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        return {"boxes": boxes_t, "scores": scores_t, "labels": labels_t}

    def _make_gt(self, boxes, labels):
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.long),
        }

    def test_perfect_detection_tp_fp_fn(self):
        """A prediction that exactly matches the GT: TP=1, FP=0, FN=0."""
        pred = [self._make_pred([[0., 0., 10., 10.]], [0])]
        gt = [self._make_gt([[0., 0., 10., 10.]], [0])]
        TP, FP, FN = detection_confusion(pred, gt)
        self.assertEqual(TP, 1)
        self.assertEqual(FP, 0)
        self.assertEqual(FN, 0)

    def test_no_detection_all_fn(self):
        """Empty predictions: all GT boxes are FN."""
        pred = [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}]
        gt = [self._make_gt([[0., 0., 10., 10.], [20., 20., 30., 30.]], [0, 1])]
        TP, FP, FN = detection_confusion(pred, gt)
        self.assertEqual(TP, 0)
        self.assertEqual(FP, 0)
        self.assertEqual(FN, 2)

    def test_all_false_positives(self):
        """Predictions with no GT: all predictions are FP."""
        pred = [self._make_pred([[0., 0., 10., 10.], [20., 20., 30., 30.]], [0, 1])]
        gt = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long),
               "iscrowd": torch.zeros(0, dtype=torch.long)}]
        TP, FP, FN = detection_confusion(pred, gt)
        self.assertEqual(TP, 0)
        self.assertEqual(FP, 2)
        self.assertEqual(FN, 0)

    def test_below_iou_threshold_is_fp_and_fn(self):
        """A prediction that barely misses the GT box (low IoU) is FP+FN."""
        pred = [self._make_pred([[0., 0., 5., 5.]], [0])]  # tiny box
        gt = [self._make_gt([[0., 0., 100., 100.]], [0])]  # large GT
        # IoU = 25/10000 << 0.5
        TP, FP, FN = detection_confusion(pred, gt, iou_threshold=0.5)
        self.assertEqual(TP, 0)
        self.assertEqual(FP, 1)
        self.assertEqual(FN, 1)

    def test_precision_recall_perfect(self):
        """Perfect detection should give precision=1 and recall=1 (via mAP50 ≈ 1)."""
        pred = [self._make_pred([[0., 0., 50., 50.]], [0], scores=[0.99])]
        gt = [self._make_gt([[0., 0., 50., 50.]], [0])]
        map50 = calculate_mAP50(pred, gt)
        self.assertAlmostEqual(map50, 1.0, delta=0.01,
                               msg=f"Perfect detection should give mAP50≈1, got {map50:.4f}")

    def test_map50_zero_on_complete_miss(self):
        """A prediction in the wrong location should give mAP50 = 0."""
        pred = [self._make_pred([[200., 200., 250., 250.]], [0], scores=[0.99])]
        gt = [self._make_gt([[0., 0., 10., 10.]], [0])]
        map50 = calculate_mAP50(pred, gt)
        self.assertAlmostEqual(map50, 0.0, delta=0.01,
                               msg=f"Complete miss should give mAP50≈0, got {map50:.4f}")

    def test_duplicate_prediction_not_double_counted(self):
        """Two identical predictions for one GT box: only one TP, the other FP."""
        pred = [self._make_pred([[0., 0., 10., 10.], [0., 0., 10., 10.]], [0, 0])]
        gt = [self._make_gt([[0., 0., 10., 10.]], [0])]
        TP, FP, FN = detection_confusion(pred, gt)
        self.assertEqual(TP, 1)
        self.assertEqual(FP, 1)
        self.assertEqual(FN, 0)

if __name__ == "__main__":
    unittest.main()
