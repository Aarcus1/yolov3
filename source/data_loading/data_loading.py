import numpy as np
from torchvision.datasets import CocoDetection

from source.data_loading.albumentation_pipeline import build_albumentations_pipeline
from source.data_loading.multi_sample_aug_dataset import MultiSampleAugDataset, AugmentationType
from torch.utils.data import DataLoader

def filter_invalid_boxes_and_targets(bboxes, category_ids, target, image_size, min_area):
    filtered = []
    filtered_cat_ids = []
    filtered_target = []

    for bbox, cat_id, obj in zip(bboxes, category_ids, target):
        x_min, y_min, x_max, y_max = bbox

        if any([not np.isfinite(coord) for coord in bbox]):
            continue

        # Non-positive dimensions
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            continue

        # Too small boxes
        if width * height < min_area:
            continue

        # Outside of image, could be cropped instead of filtering
        img_w, img_h = image_size
        if x_min < 0 or y_min < 0 or x_max > img_w or y_max > img_h:
            continue

        filtered.append(bbox)
        filtered_cat_ids.append(cat_id)
        filtered_target.append(obj)
    return filtered, filtered_cat_ids, filtered_target


def _apply_augmentation_pipeline(pipeline, image, target):
    image = np.array(image)

    if len(target) == 0:
        augmented = pipeline(image=image, bboxes=[], category_ids=[])
        return augmented["image"], target

    boxes = [obj["bbox"] for obj in target]
    category_ids = [obj["category_id"] for obj in target]

    boxes, category_ids, target = filter_invalid_boxes_and_targets(
        boxes, category_ids, target, image_size=image.shape[:2][::-1], min_area=1.0)

    augmented = pipeline(image=image, bboxes=boxes, category_ids=category_ids)

    new_target = []
    for bbox, cat_id, obj in zip(augmented["bboxes"], augmented["category_ids"], target):
        new_target.append({**obj, "bbox": list(bbox), "category_id": cat_id})

    return augmented["image"], new_target

def _collate_fn(batch):
    # batch items are (image, target, aug_type) from MultiSampleAugDataset,
    # or (image, target) from standard CocoDetection.
    if len(batch[0]) == 3:
        images, targets, _ = zip(*batch)
    else:
        images, targets = zip(*batch)
    return list(images), list(targets)

def get_coco_data_loader(dataset_config, image_size):
    augmentation_config = dataset_config["augmentation"]

    mosaic_config = augmentation_config.get("mosaic")
    mosaic_enabled = mosaic_config is not None and mosaic_config.get("enabled", False)

    mixup_config = augmentation_config.get("mixup")
    mixup_enabled = mixup_config is not None and mixup_config.get("enabled", False)

    multi_sample_enabled = mosaic_enabled or mixup_enabled

    standard_pipeline = build_albumentations_pipeline(image_size, augmentation_config, is_mosaic_enabled=False)
    multi_sample_pipeline = build_albumentations_pipeline(image_size, augmentation_config, is_mosaic_enabled=True) if multi_sample_enabled else None

    def transform_fn_multi_sample(image, target, aug_type: AugmentationType):
        pipeline = multi_sample_pipeline if aug_type == AugmentationType.MOSAIC else standard_pipeline
        return _apply_augmentation_pipeline(pipeline, image, target)

    def transform_fn_normal(image, target):
        return _apply_augmentation_pipeline(standard_pipeline, image, target)

    if multi_sample_enabled:
        base_dataset = CocoDetection(
            root=dataset_config['image_directory_path'],
            annFile=dataset_config['annotation_file_path'],
        )
        dataset = MultiSampleAugDataset(
            base_dataset=base_dataset,
            image_size=(int(image_size[0]), int(image_size[1])),
            mosaic_prob=float(mosaic_config["probability"]) if mosaic_enabled else 0.0,
            mixup_prob=float(mixup_config["probability"]) if mixup_enabled else 0.0,
            mixup_alpha=float(mixup_config["alpha"]) if mixup_enabled else 0.0,
            transform_fn=transform_fn_multi_sample,
        )
    else:
        dataset = CocoDetection(
            root=dataset_config['image_directory_path'],
            annFile=dataset_config['annotation_file_path'],
            transforms=transform_fn_normal
        )

    worker_count = dataset_config['worker_count']
    loader = DataLoader(
        dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=dataset_config['do_shuffle'],
        collate_fn=_collate_fn,
        pin_memory=dataset_config['do_pin_memory'],
        num_workers=worker_count,
        persistent_workers=worker_count > 0,
        prefetch_factor=dataset_config['prefetch_factor'] if worker_count > 0 else None
    )

    return loader
