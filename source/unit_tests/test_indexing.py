"""
Tests for spatial indexing correctness in build_yolo_targets and decode_yolo_preds.

The key invariant: a box placed at pixel coordinate (cx, cy) must end up in
grid cell (row=floor(cy/stride), col=floor(cx/stride)), and decoding that
cell must recover approximately the same (cx, cy, w, h).
"""

import unittest
import torch
from source.utilities.data_conversion import (
    build_yolo_targets,
    decode_yolo_preds,
    decode_batch_outputs,
    convert_boxes_xywh_to_xyxy,
    split_annotations_per_image,
)

def make_annotation(cx, cy, w, h, category_id=1):
    """Build a COCO-style annotation dict from a centre + size."""
    return {"bbox": [cx - w / 2, cy - h / 2, w, h], "category_id": category_id}


def _single_target(image_size, stride, anchors, strides, ann, iou_ignore_thresh=0.0):
    """Run build_yolo_targets for a single image / single annotation."""
    original_index_to_packed_index_mapping = list(range(80))  # identity mapping
    targets = build_yolo_targets(
        gt_targets=[[ann]],
        image_size=image_size,
        anchors=anchors,
        strides=strides,
        num_classes=80,
        device=torch.device("cpu"),
        iou_ignore_thresh=iou_ignore_thresh,
        original_index_to_packed_index_mapping=original_index_to_packed_index_mapping,
    )
    return targets

class TestBuildYoloTargetsIndexing(unittest.TestCase):
    """build_yolo_targets: the objectness flag must land in the correct (row, col)."""

    def setUp(self):
        # Use a single scale with one large anchor so every box matches it.
        self.image_size = (416, 416)
        self.strides = [8]
        self.anchors = [torch.tensor([[32.0, 32.0]])]  # 1 anchor

    def _get_target(self, cx, cy, w=20.0, h=20.0):
        ann = make_annotation(cx, cy, w, h)
        targets = _single_target(self.image_size, self.strides[0], self.anchors, self.strides, ann)
        # targets[0] shape: [B=1, A=1, H=52, W=52, 85]
        return targets[0][0, 0]  # [H, W, 85]

    def test_objectness_at_expected_row_col(self):
        """Objectness==1 must be at row=floor(cy/stride), col=floor(cx/stride)."""
        cx, cy = 100.0, 200.0
        stride = self.strides[0]
        expected_row = int(cy // stride)
        expected_col = int(cx // stride)

        t = self._get_target(cx, cy)
        obj = t[..., 4]  # [H, W]

        actual_positions = (obj == 1.0).nonzero(as_tuple=False)
        self.assertEqual(actual_positions.shape[0], 1, "Expected exactly one objectness=1 cell")

        actual_row, actual_col = actual_positions[0].tolist()
        self.assertEqual(actual_row, expected_row,
                         f"Row mismatch: got {actual_row}, expected {expected_row} for cy={cy}")
        self.assertEqual(actual_col, expected_col,
                         f"Col mismatch: got {actual_col}, expected {expected_col} for cx={cx}")

    def test_x_offset_stored_in_channel_0(self):
        """tx (sub-cell x offset) must be stored at [..., 0], not [..., 1]."""
        cx, cy = 100.0, 200.0
        stride = self.strides[0]
        gcol = int(cx // stride)
        grow = int(cy // stride)
        expected_tx = cx / stride - gcol
        expected_ty = cy / stride - grow

        t = self._get_target(cx, cy)
        stored_tx = t[grow, gcol, 0].item()
        stored_ty = t[grow, gcol, 1].item()

        self.assertAlmostEqual(stored_tx, expected_tx, places=5,
                               msg=f"tx stored wrong: got {stored_tx}, expected {expected_tx}")
        self.assertAlmostEqual(stored_ty, expected_ty, places=5,
                               msg=f"ty stored wrong: got {stored_ty}, expected {expected_ty}")

    def test_x_axis_varies_with_cx(self):
        """Changing cx must change the column, not the row."""
        cy = 100.0  # fixed
        stride = self.strides[0]
        fixed_row = int(cy // stride)

        for cx in [50.0, 130.0, 250.0, 370.0]:
            expected_col = int(cx // stride)
            t = self._get_target(cx, cy)
            pos = (t[..., 4] == 1.0).nonzero(as_tuple=False)
            self.assertEqual(pos.shape[0], 1)
            row, col = pos[0].tolist()
            self.assertEqual(row, fixed_row, f"cx={cx}: row should stay at {fixed_row}, got {row}")
            self.assertEqual(col, expected_col, f"cx={cx}: col should be {expected_col}, got {col}")

    def test_y_axis_varies_with_cy(self):
        """Changing cy must change the row, not the column."""
        cx = 100.0  # fixed
        stride = self.strides[0]
        fixed_col = int(cx // stride)

        for cy in [50.0, 130.0, 250.0, 370.0]:
            expected_row = int(cy // stride)
            t = self._get_target(cx, cy)
            pos = (t[..., 4] == 1.0).nonzero(as_tuple=False)
            self.assertEqual(pos.shape[0], 1)
            row, col = pos[0].tolist()
            self.assertEqual(col, fixed_col, f"cy={cy}: col should stay at {fixed_col}, got {col}")
            self.assertEqual(row, expected_row, f"cy={cy}: row should be {expected_row}, got {row}")

    def test_near_top_left_corner(self):
        """Box near (0,0) should land in cell (row=0, col=0)."""
        cx, cy = 4.0, 4.0
        t = self._get_target(cx, cy)
        pos = (t[..., 4] == 1.0).nonzero(as_tuple=False)
        self.assertEqual(pos.shape[0], 1)
        self.assertEqual(pos[0].tolist(), [0, 0])

    def test_near_bottom_right_corner(self):
        """Box near (415, 415) should land in the last grid cell."""
        stride = self.strides[0]
        H = self.image_size[0] // stride  # 52
        W = self.image_size[1] // stride  # 52
        cx, cy = 412.0, 412.0
        t = self._get_target(cx, cy)
        pos = (t[..., 4] == 1.0).nonzero(as_tuple=False)
        self.assertEqual(pos.shape[0], 1)
        row, col = pos[0].tolist()
        self.assertEqual(row, int(cy // stride))
        self.assertEqual(col, int(cx // stride))
        # Must be within grid bounds
        self.assertLess(row, H)
        self.assertLess(col, W)


class TestDecodeYoloPreds(unittest.TestCase):
    """decode_yolo_preds: decoding a hand-crafted target tensor must recover the original box."""

    def setUp(self):
        self.image_size = (416, 416)
        self.strides = [8]
        self.anchors = [torch.tensor([[32.0, 32.0]])]

    def _encode_and_decode(self, cx, cy, w, h):
        """
        Manually encode (cx, cy, w, h) into a target tensor at the correct cell,
        then decode it and return the recovered boxes.
        """
        stride = self.strides[0]
        anchor_w, anchor_h = 32.0, 32.0
        H = self.image_size[0] // stride
        W = self.image_size[1] // stride
        A = 1
        C = 80

        grow = int(cy // stride)
        gcol = int(cx // stride)
        tx = cx / stride - gcol
        ty = cy / stride - grow
        tw = torch.log(torch.tensor(w / anchor_w))
        th = torch.log(torch.tensor(h / anchor_h))

        # Build pred tensor [A, H, W, 5+C] with are_logits=False
        # (so values are used directly, no sigmoid/exp on tx/ty/obj)
        p = torch.zeros(A, H, W, 5 + C)
        p[0, grow, gcol, 0] = tx       # already in [0,1], no sigmoid needed
        p[0, grow, gcol, 1] = ty
        p[0, grow, gcol, 2] = tw
        p[0, grow, gcol, 3] = th
        p[0, grow, gcol, 4] = 1.0      # objectness
        p[0, grow, gcol, 5] = 1.0      # class 0

        boxes_list, obj_scores_list, _, _ = decode_yolo_preds(
            preds=[p],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            are_logits=False,
        )

        # boxes are flat [A*H*W, 4]; find the one with high objectness
        obj_scores = obj_scores_list[0]
        boxes = boxes_list[0]
        best_idx = obj_scores.argmax().item()
        return boxes[best_idx]

    def test_decode_recovers_center(self):
        """Decoded box center must match the original (cx, cy)."""
        cx, cy, w, h = 100.0, 200.0, 40.0, 60.0
        box = self._encode_and_decode(cx, cy, w, h)
        x1, y1, x2, y2 = box.tolist()
        decoded_cx = (x1 + x2) / 2
        decoded_cy = (y1 + y2) / 2
        self.assertAlmostEqual(decoded_cx, cx, places=3, msg=f"cx mismatch: {decoded_cx} vs {cx}")
        self.assertAlmostEqual(decoded_cy, cy, places=3, msg=f"cy mismatch: {decoded_cy} vs {cy}")

    def test_decode_recovers_size(self):
        """Decoded box width/height must match the original w, h."""
        cx, cy, w, h = 100.0, 200.0, 40.0, 60.0
        box = self._encode_and_decode(cx, cy, w, h)
        x1, y1, x2, y2 = box.tolist()
        decoded_w = x2 - x1
        decoded_h = y2 - y1
        self.assertAlmostEqual(decoded_w, w, places=3, msg=f"w mismatch: {decoded_w} vs {w}")
        self.assertAlmostEqual(decoded_h, h, places=3, msg=f"h mismatch: {decoded_h} vs {h}")

    def test_decode_x_not_confused_with_y(self):
        """
        Use an asymmetric (cx != cy) box and verify the decoded cx/cy are not swapped.
        If grid axes were swapped, cx and cy would be exchanged.
        """
        cx, cy, w, h = 80.0, 240.0, 20.0, 20.0  # clearly asymmetric
        box = self._encode_and_decode(cx, cy, w, h)
        x1, y1, x2, y2 = box.tolist()
        decoded_cx = (x1 + x2) / 2
        decoded_cy = (y1 + y2) / 2
        self.assertAlmostEqual(decoded_cx, cx, places=3)
        self.assertAlmostEqual(decoded_cy, cy, places=3)
        # Explicitly assert they are NOT swapped
        self.assertNotAlmostEqual(decoded_cx, cy, places=0,
                                  msg="cx and cy appear to be swapped in decoder")

    def test_clamping_respects_width_height(self):
        """
        A box extending past the right edge should have x2 clamped to img_w,
        and a box extending past the bottom should have y2 clamped to img_h.
        """
        img_h, img_w = self.image_size
        # Box centred near the right edge, wide enough to overflow
        cx, cy, w, h = 410.0, 208.0, 64.0, 20.0
        box = self._encode_and_decode(cx, cy, w, h)
        _, _, x2, y2 = box.tolist()
        self.assertLessEqual(x2, img_w + 1e-4, "x2 not clamped to image width")
        self.assertLessEqual(y2, img_h + 1e-4, "y2 not clamped to image height")


class TestRoundTrip(unittest.TestCase):
    """
    End-to-end: build_yolo_targets → decode_yolo_preds must recover the
    original box coordinates (within sub-pixel rounding from int(center/stride)).
    """

    def setUp(self):
        self.image_size = (416, 416)
        self.strides = [8, 16, 32]
        self.anchors = [
            torch.tensor([[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]]),
            torch.tensor([[30.0, 61.0], [62.0, 45.0], [59.0, 119.0]]),
            torch.tensor([[116.0, 90.0], [156.0, 198.0], [373.0, 326.0]]),
        ]
        self.identity_mapping = list(range(80))

    def _round_trip(self, cx, cy, w, h, category_id=1):
        ann = make_annotation(cx, cy, w, h, category_id)
        targets = build_yolo_targets(
            gt_targets=[[ann]],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            num_classes=80,
            device=torch.device("cpu"),
            iou_ignore_thresh=0.0,
            original_index_to_packed_index_mapping=self.identity_mapping,
        )
        # Decode as if these are non-logit targets (values already decoded)
        boxes_list, obj_scores_list, class_scores_list, labels_list = decode_yolo_preds(
            preds=[t[0] for t in targets],   # drop batch dim → [A, H, W, 85]
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            are_logits=False,
        )
        # Find the cell with objectness == 1.0 across all scales
        all_boxes = torch.cat(boxes_list, dim=0)
        all_obj = torch.cat(obj_scores_list, dim=0)
        best_idx = all_obj.argmax().item()
        return all_boxes[best_idx], all_obj[best_idx].item()

    def test_small_box_round_trip(self):
        cx, cy, w, h = 100.0, 200.0, 20.0, 30.0
        box, obj = self._round_trip(cx, cy, w, h)
        x1, y1, x2, y2 = box.tolist()
        self.assertAlmostEqual((x1 + x2) / 2, cx, delta=8.0,
                               msg="cx not recovered in round-trip")
        self.assertAlmostEqual((y1 + y2) / 2, cy, delta=8.0,
                               msg="cy not recovered in round-trip")
        self.assertAlmostEqual(x2 - x1, w, delta=1.0)
        self.assertAlmostEqual(y2 - y1, h, delta=1.0)

    def test_large_box_round_trip(self):
        cx, cy, w, h = 208.0, 208.0, 200.0, 160.0
        box, obj = self._round_trip(cx, cy, w, h)
        x1, y1, x2, y2 = box.tolist()
        self.assertAlmostEqual((x1 + x2) / 2, cx, delta=32.0)
        self.assertAlmostEqual((y1 + y2) / 2, cy, delta=32.0)
        self.assertAlmostEqual(x2 - x1, w, delta=1.0)
        self.assertAlmostEqual(y2 - y1, h, delta=1.0)

    def test_axes_not_swapped_in_round_trip(self):
        """Asymmetric box: decoded cx/cy must not be swapped."""
        cx, cy, w, h = 60.0, 300.0, 30.0, 30.0  # very different x vs y
        box, _ = self._round_trip(cx, cy, w, h)
        x1, y1, x2, y2 = box.tolist()
        decoded_cx = (x1 + x2) / 2
        decoded_cy = (y1 + y2) / 2
        # decoded_cx should be close to 60, not 300
        self.assertLess(abs(decoded_cx - cx), abs(decoded_cx - cy),
                        msg=f"cx and cy appear swapped: decoded ({decoded_cx:.1f}, {decoded_cy:.1f}), "
                            f"expected ({cx}, {cy})")


class TestAnchorOrdering(unittest.TestCase):
    """
    Anchors are stored as [anchor_w, anchor_h].
    tw = log(box_w / anchor_w) and th = log(box_h / anchor_h).
    If w and h were swapped in the anchor tensor the decoded size would be wrong.
    """

    def setUp(self):
        self.image_size = (416, 416)
        self.strides = [8]
        self.identity_mapping = list(range(80))

    def _round_trip_size(self, w, h, anchor_w, anchor_h):
        anchors = [torch.tensor([[anchor_w, anchor_h]])]
        ann = make_annotation(cx=208.0, cy=208.0, w=w, h=h)
        targets = build_yolo_targets(
            gt_targets=[[ann]],
            image_size=self.image_size,
            anchors=anchors,
            strides=self.strides,
            num_classes=80,
            device=torch.device("cpu"),
            iou_ignore_thresh=0.0,
            original_index_to_packed_index_mapping=self.identity_mapping,
        )
        boxes_list, obj_scores_list, _, _ = decode_yolo_preds(
            preds=[targets[0][0]],
            image_size=self.image_size,
            anchors=anchors,
            strides=self.strides,
            are_logits=False,
        )
        all_boxes = boxes_list[0]
        all_obj = obj_scores_list[0]
        best = all_obj.argmax().item()
        x1, y1, x2, y2 = all_boxes[best].tolist()
        return x2 - x1, y2 - y1  # decoded_w, decoded_h

    def test_anchor_w_h_not_swapped_square(self):
        """With a square box and square anchor there is no swapping to detect; use asymmetric ones."""
        decoded_w, decoded_h = self._round_trip_size(w=40.0, h=80.0,
                                                     anchor_w=32.0, anchor_h=64.0)
        self.assertAlmostEqual(decoded_w, 40.0, delta=0.5,
                               msg=f"decoded_w={decoded_w:.2f}, expected 40.0 — anchor_w/h may be swapped")
        self.assertAlmostEqual(decoded_h, 80.0, delta=0.5,
                               msg=f"decoded_h={decoded_h:.2f}, expected 80.0 — anchor_w/h may be swapped")

    def test_anchor_w_h_not_swapped_tall_box(self):
        """A tall box (h >> w) must stay tall after round-trip."""
        decoded_w, decoded_h = self._round_trip_size(w=20.0, h=120.0,
                                                     anchor_w=16.0, anchor_h=96.0)
        self.assertAlmostEqual(decoded_w, 20.0, delta=0.5)
        self.assertAlmostEqual(decoded_h, 120.0, delta=0.5)
        # If swapped, decoded_w would be ~120 and decoded_h ~20
        self.assertLess(decoded_w, decoded_h,
                        msg="Box should still be taller than wide after round-trip")

    def test_anchor_w_h_not_swapped_wide_box(self):
        """A wide box (w >> h) must stay wide after round-trip."""
        decoded_w, decoded_h = self._round_trip_size(w=120.0, h=20.0,
                                                     anchor_w=96.0, anchor_h=16.0)
        self.assertAlmostEqual(decoded_w, 120.0, delta=0.5)
        self.assertAlmostEqual(decoded_h, 20.0, delta=0.5)
        self.assertGreater(decoded_w, decoded_h,
                           msg="Box should still be wider than tall after round-trip")


class TestTwThEncoding(unittest.TestCase):
    """
    tw = log(box_w / anchor_w) stored in channel 2,
    th = log(box_h / anchor_h) stored in channel 3.
    Verify the correct ratio ends up in the correct channel.
    """

    def setUp(self):
        self.image_size = (416, 416)
        self.strides = [8]
        self.identity_mapping = list(range(80))

    def test_tw_th_channels_not_swapped(self):
        """tw (width log-ratio) must be in channel 2, th in channel 3."""
        anchor_w, anchor_h = 32.0, 64.0
        box_w, box_h = 48.0, 16.0  # asymmetric: tw = log(48/32)>0, th = log(16/64)<0
        anchors = [torch.tensor([[anchor_w, anchor_h]])]
        ann = make_annotation(cx=208.0, cy=208.0, w=box_w, h=box_h)
        targets = build_yolo_targets(
            gt_targets=[[ann]],
            image_size=self.image_size,
            anchors=anchors,
            strides=self.strides,
            num_classes=80,
            device=torch.device("cpu"),
            iou_ignore_thresh=0.0,
            original_index_to_packed_index_mapping=self.identity_mapping,
        )
        t = targets[0][0, 0]  # [H, W, 85]
        obj_pos = (t[..., 4] == 1.0).nonzero(as_tuple=False)[0]
        grow, gcol = obj_pos.tolist()

        stored_tw = t[grow, gcol, 2].item()
        stored_th = t[grow, gcol, 3].item()

        expected_tw = torch.log(torch.tensor(box_w / anchor_w)).item()
        expected_th = torch.log(torch.tensor(box_h / anchor_h)).item()

        self.assertAlmostEqual(stored_tw, expected_tw, places=4,
                               msg=f"tw mismatch: stored {stored_tw:.4f}, expected {expected_tw:.4f}")
        self.assertAlmostEqual(stored_th, expected_th, places=4,
                               msg=f"th mismatch: stored {stored_th:.4f}, expected {expected_th:.4f}")

        # tw and th must have opposite signs for this box
        self.assertGreater(stored_tw, 0, "tw should be positive (box wider than anchor)")
        self.assertLess(stored_th, 0, "th should be negative (box shorter than anchor)")


class TestClassLabelIndexing(unittest.TestCase):
    """
    category_id is 1-based in COCO. The mapping converts it to a packed 0-based index.
    Verify the correct one-hot channel is set in the target tensor.
    """

    def setUp(self):
        self.image_size = (416, 416)
        self.strides = [8]
        self.anchors = [torch.tensor([[32.0, 32.0]])]
        # COCO-like non-contiguous ids: 1,2,3 → packed 0,1,2; id 4 skipped; 5 → 3
        self.ids = [1, 2, 3, 5]
        from source.utilities.utilities import convert_to_zero_indexed_list, invert_index_list
        zero_indexed = convert_to_zero_indexed_list(self.ids)
        self.mapping = invert_index_list(zero_indexed)

    def _get_class_channel(self, category_id):
        ann = make_annotation(cx=208.0, cy=208.0, w=20.0, h=20.0, category_id=category_id)
        targets = build_yolo_targets(
            gt_targets=[[ann]],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            num_classes=4,
            device=torch.device("cpu"),
            iou_ignore_thresh=0.0,
            original_index_to_packed_index_mapping=self.mapping,
        )
        t = targets[0][0, 0]  # [H, W, 9]
        obj_pos = (t[..., 4] == 1.0).nonzero(as_tuple=False)[0]
        grow, gcol = obj_pos.tolist()
        class_channels = t[grow, gcol, 5:]  # [num_classes]
        hot = (class_channels == 1.0).nonzero(as_tuple=False)
        self.assertEqual(hot.shape[0], 1, "Expected exactly one hot class channel")
        return hot[0, 0].item()

    def test_first_id_maps_to_channel_0(self):
        self.assertEqual(self._get_class_channel(category_id=1), 0)

    def test_second_id_maps_to_channel_1(self):
        self.assertEqual(self._get_class_channel(category_id=2), 1)

    def test_non_contiguous_id_maps_correctly(self):
        """category_id=5 skips id=4, should map to packed index 3."""
        self.assertEqual(self._get_class_channel(category_id=5), 3)

    def test_one_hot_has_exactly_one_active(self):
        """Only one class channel should be 1.0 per annotation."""
        ann = make_annotation(cx=208.0, cy=208.0, w=20.0, h=20.0, category_id=2)
        targets = build_yolo_targets(
            gt_targets=[[ann]],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            num_classes=4,
            device=torch.device("cpu"),
            iou_ignore_thresh=0.0,
            original_index_to_packed_index_mapping=self.mapping,
        )
        t = targets[0][0, 0]
        class_channels = t[..., 5:]  # entire grid
        self.assertEqual((class_channels == 1.0).sum().item(), 1,
                         "Exactly one class channel should be hot across the whole grid")


class TestNonSquareImageIndexing(unittest.TestCase):
    """
    With a non-square image (H != W), row/col confusion that happens to be hidden
    by symmetry on square images will surface here.
    """

    def setUp(self):
        # Tall image: more rows than columns
        self.image_size = (480, 320)  # H=480, W=320
        self.strides = [16]
        self.anchors = [torch.tensor([[32.0, 32.0]])]
        self.identity_mapping = list(range(80))

    def test_grid_shape_matches_hw(self):
        """Target tensor spatial dims must be (H//stride, W//stride), not (W//stride, H//stride)."""
        img_h, img_w = self.image_size
        stride = self.strides[0]
        expected_H = img_h // stride  # 30
        expected_W = img_w // stride  # 20

        ann = make_annotation(cx=160.0, cy=240.0, w=30.0, h=30.0)
        targets = build_yolo_targets(
            gt_targets=[[ann]],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            num_classes=80,
            device=torch.device("cpu"),
            iou_ignore_thresh=0.0,
            original_index_to_packed_index_mapping=self.identity_mapping,
        )
        _, _, H, W, _ = targets[0].shape
        self.assertEqual(H, expected_H, f"H should be {expected_H}, got {H}")
        self.assertEqual(W, expected_W, f"W should be {expected_W}, got {W}")

    def test_row_col_correct_on_non_square(self):
        """On a non-square image, row=floor(cy/stride) and col=floor(cx/stride) must hold."""
        cx, cy = 160.0, 240.0
        stride = self.strides[0]
        expected_row = int(cy // stride)  # 15
        expected_col = int(cx // stride)  # 10

        ann = make_annotation(cx=cx, cy=cy, w=30.0, h=30.0)
        targets = build_yolo_targets(
            gt_targets=[[ann]],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            num_classes=80,
            device=torch.device("cpu"),
            iou_ignore_thresh=0.0,
            original_index_to_packed_index_mapping=self.identity_mapping,
        )
        t = targets[0][0, 0]  # [H, W, 85]
        pos = (t[..., 4] == 1.0).nonzero(as_tuple=False)
        self.assertEqual(pos.shape[0], 1)
        actual_row, actual_col = pos[0].tolist()
        self.assertEqual(actual_row, expected_row,
                         f"row: got {actual_row}, expected {expected_row}")
        self.assertEqual(actual_col, expected_col,
                         f"col: got {actual_col}, expected {expected_col}")

    def test_decode_recovers_position_non_square(self):
        """Round-trip on a non-square image must recover (cx, cy) with axes not swapped."""
        cx, cy, w, h = 160.0, 400.0, 30.0, 30.0  # cy >> cx
        ann = make_annotation(cx=cx, cy=cy, w=w, h=h)
        targets = build_yolo_targets(
            gt_targets=[[ann]],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            num_classes=80,
            device=torch.device("cpu"),
            iou_ignore_thresh=0.0,
            original_index_to_packed_index_mapping=self.identity_mapping,
        )
        boxes_list, obj_scores_list, _, _ = decode_yolo_preds(
            preds=[targets[0][0]],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            are_logits=False,
        )
        best = obj_scores_list[0].argmax().item()
        x1, y1, x2, y2 = boxes_list[0][best].tolist()
        decoded_cx = (x1 + x2) / 2
        decoded_cy = (y1 + y2) / 2
        self.assertAlmostEqual(decoded_cx, cx, delta=self.strides[0],
                               msg=f"cx swapped? decoded ({decoded_cx:.1f}, {decoded_cy:.1f})")
        self.assertAlmostEqual(decoded_cy, cy, delta=self.strides[0])


class TestConvertBoxesXywhToXyxy(unittest.TestCase):
    """convert_boxes_xywh_to_xyxy: verify x/y conversion and the normalize scale order."""

    def test_basic_conversion(self):
        boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0]])  # x,y,w,h
        result = convert_boxes_xywh_to_xyxy(boxes, image_size=(416, 416), normalize=False)
        x1, y1, x2, y2 = result[0].tolist()
        self.assertAlmostEqual(x1, 10.0)
        self.assertAlmostEqual(y1, 20.0)
        self.assertAlmostEqual(x2, 60.0)   # x + w
        self.assertAlmostEqual(y2, 100.0)  # y + h

    def test_x2_uses_width_not_height(self):
        """x2 = x + w, y2 = y + h — they must not be swapped."""
        boxes = torch.tensor([[0.0, 0.0, 30.0, 70.0]])  # w=30, h=70
        result = convert_boxes_xywh_to_xyxy(boxes, image_size=(416, 416), normalize=False)
        x1, y1, x2, y2 = result[0].tolist()
        self.assertAlmostEqual(x2 - x1, 30.0, msg="x2-x1 should equal w=30, not h=70")
        self.assertAlmostEqual(y2 - y1, 70.0, msg="y2-y1 should equal h=70, not w=30")

    def test_normalize_uses_correct_scale_per_axis(self):
        """
        With image_size=(H, W)=(200, 400):
          x-coords must divide by W=400, y-coords by H=200.
        The bug would divide x by H and y by W.
        """
        # Box at x=0, y=0, w=400, h=200 → after correct normalisation x2=1.0, y2=1.0
        boxes = torch.tensor([[0.0, 0.0, 400.0, 200.0]])
        image_size = (200, 400)  # H=200, W=400
        result = convert_boxes_xywh_to_xyxy(boxes, image_size=image_size, normalize=True)
        x1, y1, x2, y2 = result[0].tolist()
        self.assertAlmostEqual(x2, 1.0, places=5,
                               msg=f"x2 should normalise to 1.0 with W=400, got {x2} "
                                   f"(if ~2.0 then H and W are swapped in normalize)")
        self.assertAlmostEqual(y2, 1.0, places=5,
                               msg=f"y2 should normalise to 1.0 with H=200, got {y2}")

    def test_normalize_asymmetric_box(self):
        """x and y normalise by different factors on a non-square image."""
        image_size = (100, 200)  # H=100, W=200
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]])  # square box
        result = convert_boxes_xywh_to_xyxy(boxes, image_size=image_size, normalize=True)
        x1, y1, x2, y2 = result[0].tolist()
        # x2 = 100/W = 100/200 = 0.5
        # y2 = 100/H = 100/100 = 1.0
        self.assertAlmostEqual(x2, 0.5, places=5,
                               msg="x2 should be 0.5 (100/W=200)")
        self.assertAlmostEqual(y2, 1.0, places=5,
                               msg="y2 should be 1.0 (100/H=100)")


class TestSplitAnnotationsPerImage(unittest.TestCase):
    """split_annotations_per_image: boxes must be assigned to the correct image slot."""

    def test_correct_partition(self):
        from source.utilities.data_conversion import split_annotations_per_image
        # 3 images: 2, 0, 3 annotations respectively
        boxes = torch.tensor([
            [0., 0., 10., 10.],
            [1., 1., 11., 11.],   # image 0
            # image 1 has 0 boxes
            [2., 2., 12., 12.],
            [3., 3., 13., 13.],
            [4., 4., 14., 14.],   # image 2
        ])
        labels = torch.tensor([0, 1, 2, 3, 4])
        iscrowd = torch.zeros(5, dtype=torch.int64)
        num_anns = [2, 0, 3]

        results = split_annotations_per_image(boxes, labels, iscrowd, num_anns)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["boxes"].shape[0], 2)
        self.assertEqual(results[1]["boxes"].shape[0], 0)
        self.assertEqual(results[2]["boxes"].shape[0], 3)

        # Verify the actual box values, not just counts
        self.assertTrue(torch.allclose(results[0]["boxes"][0], boxes[0]))
        self.assertTrue(torch.allclose(results[0]["boxes"][1], boxes[1]))
        self.assertTrue(torch.allclose(results[2]["boxes"][0], boxes[2]))

    def test_labels_follow_boxes(self):
        """Labels must stay aligned with their corresponding boxes after partitioning."""
        from source.utilities.data_conversion import split_annotations_per_image
        boxes = torch.zeros(4, 4)
        labels = torch.tensor([10, 20, 30, 40])
        iscrowd = torch.zeros(4, dtype=torch.int64)
        results = split_annotations_per_image(boxes, labels, iscrowd, [1, 2, 1])

        self.assertEqual(results[0]["labels"][0].item(), 10)
        self.assertEqual(results[1]["labels"][0].item(), 20)
        self.assertEqual(results[1]["labels"][1].item(), 30)
        self.assertEqual(results[2]["labels"][0].item(), 40)


class TestDecodeBoxesChannelOrder(unittest.TestCase):
    """
    In the decoded flat boxes tensor, column order must be [x1, y1, x2, y2].
    Verify that column 0 < column 2 (x1 < x2) and column 1 < column 3 (y1 < y2)
    for a normal box, and that x-coords are not in the y slots.
    """

    def setUp(self):
        self.image_size = (416, 416)
        self.strides = [8]
        self.anchors = [torch.tensor([[32.0, 32.0]])]

    def test_box_column_order_is_x1y1x2y2(self):
        cx, cy, w, h = 200.0, 100.0, 40.0, 80.0  # asymmetric: cx > cy
        ann = make_annotation(cx, cy, w, h)
        targets = build_yolo_targets(
            gt_targets=[[ann]],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            num_classes=80,
            device=torch.device("cpu"),
            iou_ignore_thresh=0.0,
            original_index_to_packed_index_mapping=list(range(80)),
        )
        boxes_list, obj_scores_list, _, _ = decode_yolo_preds(
            preds=[targets[0][0]],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            are_logits=False,
        )
        best = obj_scores_list[0].argmax().item()
        box = boxes_list[0][best]
        x1, y1, x2, y2 = box.tolist()

        self.assertLess(x1, x2, "x1 must be less than x2")
        self.assertLess(y1, y2, "y1 must be less than y2")

        # cx > cy, so the x-coords should be larger than the y-coords
        decoded_cx = (x1 + x2) / 2
        decoded_cy = (y1 + y2) / 2
        self.assertGreater(decoded_cx, decoded_cy,
                           msg=f"cx={decoded_cx:.1f} should be > cy={decoded_cy:.1f} "
                               f"since input cx={cx} > cy={cy}")

    def test_width_from_channel_0_2_not_1_3(self):
        """The width (x2-x1) must come from columns 0 and 2, not 1 and 3."""
        cx, cy, w, h = 208.0, 208.0, 60.0, 20.0  # w >> h
        ann = make_annotation(cx, cy, w, h)
        targets = build_yolo_targets(
            gt_targets=[[ann]],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            num_classes=80,
            device=torch.device("cpu"),
            iou_ignore_thresh=0.0,
            original_index_to_packed_index_mapping=list(range(80)),
        )
        boxes_list, obj_scores_list, _, _ = decode_yolo_preds(
            preds=[targets[0][0]],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            are_logits=False,
        )
        best = obj_scores_list[0].argmax().item()
        x1, y1, x2, y2 = boxes_list[0][best].tolist()
        decoded_w = x2 - x1
        decoded_h = y2 - y1
        self.assertGreater(decoded_w, decoded_h,
                           msg=f"Decoded box should be wider (w={decoded_w:.1f}) than tall "
                               f"(h={decoded_h:.1f}); original w={w}, h={h}")


class TestDecodeFullBatch(unittest.TestCase):
    """
    decode_batch_outputs iterates over batch indices and slices out[b].
    Each image in the batch must decode independently and correctly.
    """

    def setUp(self):
        self.image_size = (416, 416)
        self.strides = [8]
        self.anchors = [torch.tensor([[32.0, 32.0]])]
        self.identity_mapping = list(range(80))

    def test_batch_images_decode_independently(self):
        """Two images with boxes at different locations must decode to those locations."""
        from source.utilities.data_conversion import decode_batch_outputs

        cx0, cy0 = 50.0, 50.0    # image 0: top-left region
        cx1, cy1 = 350.0, 350.0  # image 1: bottom-right region

        ann0 = make_annotation(cx0, cy0, 30.0, 30.0)
        ann1 = make_annotation(cx1, cy1, 30.0, 30.0)

        targets = build_yolo_targets(
            gt_targets=[[ann0], [ann1]],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            num_classes=80,
            device=torch.device("cpu"),
            iou_ignore_thresh=0.0,
            original_index_to_packed_index_mapping=self.identity_mapping,
        )

        decoded = decode_batch_outputs(
            outputs=targets,
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            are_logits=False,
        )

        self.assertEqual(len(decoded), 2)

        for img_idx, (expected_cx, expected_cy) in enumerate([(cx0, cy0), (cx1, cy1)]):
            boxes_list, obj_scores_list, _, _ = decoded[img_idx]
            all_boxes = torch.cat(boxes_list, dim=0)
            all_obj = torch.cat(obj_scores_list, dim=0)
            best = all_obj.argmax().item()
            x1, y1, x2, y2 = all_boxes[best].tolist()
            decoded_cx = (x1 + x2) / 2
            decoded_cy = (y1 + y2) / 2
            stride = self.strides[0]
            self.assertAlmostEqual(decoded_cx, expected_cx, delta=stride,
                                   msg=f"Image {img_idx}: cx mismatch")
            self.assertAlmostEqual(decoded_cy, expected_cy, delta=stride,
                                   msg=f"Image {img_idx}: cy mismatch")

    def test_batch_index_not_mixed(self):
        """Box from image 0 must NOT appear in image 1's decoded output."""
        cx0, cy0 = 50.0, 50.0    # clearly separate locations
        cx1, cy1 = 350.0, 350.0

        ann0 = make_annotation(cx0, cy0, 30.0, 30.0)
        ann1 = make_annotation(cx1, cy1, 30.0, 30.0)

        targets = build_yolo_targets(
            gt_targets=[[ann0], [ann1]],
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            num_classes=80,
            device=torch.device("cpu"),
            iou_ignore_thresh=0.0,
            original_index_to_packed_index_mapping=self.identity_mapping,
        )

        decoded = decode_batch_outputs(
            outputs=targets,
            image_size=self.image_size,
            anchors=self.anchors,
            strides=self.strides,
            are_logits=False,
        )

        # The highest-objectness box in image 1 must be near (cx1, cy1), not (cx0, cy0)
        boxes_list, obj_scores_list, _, _ = decoded[1]
        all_boxes = torch.cat(boxes_list, dim=0)
        all_obj = torch.cat(obj_scores_list, dim=0)
        best = all_obj.argmax().item()
        x1, y1, x2, y2 = all_boxes[best].tolist()
        decoded_cx = (x1 + x2) / 2
        decoded_cy = (y1 + y2) / 2

        dist_correct = ((decoded_cx - cx1) ** 2 + (decoded_cy - cy1) ** 2) ** 0.5
        dist_wrong   = ((decoded_cx - cx0) ** 2 + (decoded_cy - cy0) ** 2) ** 0.5
        self.assertLess(dist_correct, dist_wrong,
                        msg=f"Image 1 decoded to ({decoded_cx:.1f},{decoded_cy:.1f}), "
                            f"closer to image-0 box ({cx0},{cy0}) than image-1 box ({cx1},{cy1})")


if __name__ == "__main__":
    unittest.main()

