# YOLOv3 Implementation

A PyTorch implementation of the YOLOv3 object detection model, inspired by the original paper [*YOLOv3: An Incremental Improvement*](https://arxiv.org/abs/1804.02767).

## Features

Implemented from scratch in PyTorch, this repository provides a complete training and evaluation pipeline for YOLOv3 on the COCO 2017 dataset.

**Data Augmentation:**
The following augmentations are supported:
- **Mosaic** (4-image stitching) augmentation
- **MixUp** (pixel blending) augmentation
- **Random affine transforms**, horizontal flips, color jitter, and random gamma adjustments

**Model Architecture:**
A darknet-53 backbone is required and can be loaded from the original Darknet weights. The model supports multi-scale detection with 3 prediction heads (strides 8, 16, 32) and uses anchor-based target encoding with IoU-based ignore masking.

**Loss Function:**
The loss tries to resemble the original C++ implementation, with separate BCE losses for objectness and classification, and MSE for bounding box regression. The coordinate loss is weighted to emphasize small objects.

**Post-processing:**
Class-based NMS is available for post-processing detections.

**Training:**
Warmup, cosine annealing scheduler, gradient clipping, and backbone freezing are supported during training. Checkpoint management with auto-resume is implemented, and TensorBoard logging is available for metrics and losses.

**Evaluation & Testing:**
The evaluation metrics include mAP, mAP50, and mAP75 via `torchmetrics`, and sample prediction visualizations can be generated during validation. Unit tests are provided for the main pipeline components.

## Project Structure

```
yolov3/
├── configurations/          # YAML configs and Darknet .cfg
│   ├── darknet.cfg
│   └── main_config.yaml
├── data/                    # COCO 2017 dataset (images + annotations)
├── source/
│   ├── train.py             # Training entry point
│   ├── test.py              # Testing/evaluation entry point
│   ├── data_loading/        # COCO loaders, augmentation pipeline
│   ├── model/               # Darknet backbone, YOLO model
│   ├── utilities/           # Trainer, Tester, loss, metrics
│   ├── postprocess/         # NMS implementation
│   ├── serialization/       # Checkpoint save/load
│   ├── my_logging/          # TensorBoard + tqdm logger
│   ├── sampling/            # Sample prediction visualizations
│   ├── program_options/     # CLI argument parsing
│   └── unit_tests/          # Pipeline / loss / indexing tests
├── weights/backbone/        # Pretrained Darknet weights
├── checkpoints/             # Saved model checkpoints
├── runs/                    # TensorBoard logs
├── samples/                 # Saved prediction visualizations
├── requirements.txt
├── training_script.sh       # Auto-restart training wrapper
└── README.md
```

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Backbone Weights

Download the pretrained Darknet-53 backbone:

- **Config**: ./configurations/darknet.cfg
- **Weights**: [darknet.weights](https://pjreddie.com/media/files/darknet53.conv.74)

Place them at:
```
weights/backbone/darknet.weights
configurations/darknet.cfg
```

### Dataset

Download the [COCO 2017 dataset](https://cocodataset.org/#download) and organize it as:

```
data/coco_2017/
├── annotations/
│   ├── instances_training.json
│   └── instances_validation.json
└── images/
    ├── training/
    ├── validation/
    └── testing/
```

You can also generate a smaller subset with `source/data_loading/subset_extractor.py` and a matching annotation file with `source/data_loading/coco_class_extractor.py`.

## Configuration

All behavior is driven by `configurations/main_config.yaml`. Key sections:

- **`general`** — determinism mode, random seed, experiment name
- **`training`** — optimizer, scheduler, warmup, loss weights, gradient clipping, AMP, checkpoint settings, backbone freezing
- **`testing`** — checkpoint selection, save directory
- **`model`** — anchors, strides, backbone config path
- **`processing`** — NMS thresholds, YOLO IoU ignore threshold
- **`dataset`** — image size, class names/IDs, per-split paths, batch size, workers, augmentation per split

> The `dataset.ids` list maps COCO category IDs (which are non-sequential, 1–90 with gaps) to network output indices 0–79. This is required because COCO defines 91 IDs but only 80 have annotations.

## Usage

### Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Training

```bash
python source/train.py --config_folder ./configurations/
```

Or use the auto-restart wrapper (retries on crash, up to 1000 times):
```bash
./training_script.sh
```

Training will:
1. Load Darknet-53 backbone weights (optionally freeze for the first N epochs)
2. Resume from latest checkpoint if `checkpoints.mode: latest`
3. Run training + periodic validation
4. Save checkpoints every `save_interval_epochs`
5. Save sample predictions during validation
6. Log metrics to TensorBoard

### Testing / Evaluation

```bash
python source/test.py --config_folder ./configurations/
```

Runs inference on the test split, computes coco detection metrics, and saves sample prediction visualizations.

### TensorBoard

```bash
tensorboard --logdir runs/
```

Then open `http://localhost:6006`.

### Unit Tests

```bash
python -m unittest discover source/unit_tests/
```

Tests cover the loss function, target encoding/indexing, full pipeline integration, and component-level behavior.

## Architecture Notes

### Model

- **Backbone**: Darknet-53 (residual conv network) parsed from `darknet.cfg`
- **Detection heads**: 3 scales producing tensors of shape `[B, A=3, H, W, 5+C]` where `5 = (tx, ty, tw, th, tobj)` and `C` = number of classes
- **Anchors**: 9 anchors total (3 per scale), pre-defined in config

### Target Encoding

For each ground-truth box:
1. The best anchor across all scales is selected (highest wh-IoU)
2. The assigned anchor's grid cell is marked positive (`tobj=1`)
3. High-IoU non-assigned anchors are marked **ignored** (`tobj=-1`) so they aren't penalized as background
4. `tx, ty` are stored as in-cell offsets; `tw, th` as `log(box/anchor)` ratios

### Loss

Three components, summed:
- **Coordinate loss** — MSE on `xy` (after sigmoid) and `wh` (in log space), weighted by `2 - w*h` to emphasize small boxes
- **Objectness loss** — BCE with logits, separate weights for object / no-object cells, ignored cells masked out
- **Classification loss** — BCE with logits (multi-label), applied only at positive cells

### Inference

1. Model outputs raw logits
2. `decode_yolo_preds` converts them to absolute box coordinates (sigmoid for xy, exp for wh)
3. `apply_nms` filters by confidence threshold and removes overlapping boxes per class

## References

- [YOLOv3 paper](https://arxiv.org/abs/1804.02767) — Redmon & Farhadi, 2018
- [Original Darknet implementation](https://github.com/pjreddie/darknet)