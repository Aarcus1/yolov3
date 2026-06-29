import albumentations as A
from albumentations.pytorch import ToTensorV2


def _bbox_params():
    return A.BboxParams(
        format="coco",
        label_fields=["category_ids"],
        min_visibility=0.2,
    )

def _get_augmentation_config(config_root, augmentation_name: str):
    augmentation_config = config_root.get(augmentation_name)
    if augmentation_config and augmentation_config.get("enabled", False):
        return augmentation_config
    return None

def build_albumentations_pipeline(image_size, augmentation_config, is_mosaic_enabled: bool = False):
    new_w, new_h = image_size
    transforms = []

    if horizontal_flip_config := _get_augmentation_config(augmentation_config, "horizontal_flip"):
        transforms.append(A.HorizontalFlip(p=horizontal_flip_config["probability"]))

    if affine_config := _get_augmentation_config(augmentation_config, "affine"):
        if is_mosaic_enabled:
            # Mosaic's random centre point already acts as implicit translation and scale variation
            scale_limit = affine_config["scale_limit"] / 2
            shift_limit = affine_config["shift_limit"] / 2
        else:
            scale_limit = affine_config["scale_limit"]
            shift_limit = affine_config["shift_limit"]
        transforms.append(
            A.Affine(
                translate_percent=(-shift_limit, shift_limit),
                scale=(1 - scale_limit, 1 + scale_limit),
                rotate=(-affine_config["rotate_limit"], affine_config["rotate_limit"]),
                p=affine_config["probability"]
            )
        )

    transforms.append(A.Resize(height=new_h, width=new_w))

    if random_gamma_config := _get_augmentation_config(augmentation_config, "random_gamma"):
        transforms.append(
            A.RandomGamma(
                gamma_limit=tuple(random_gamma_config["gamma_limit"]),
                p=random_gamma_config["probability"],
            )
        )

    if color_jitter_config := _get_augmentation_config(augmentation_config, "color_jitter"):
        transforms.append(
            A.ColorJitter(
                brightness=color_jitter_config["brightness"],
                contrast=color_jitter_config["contrast"],
                saturation=color_jitter_config["saturation"],
                hue=color_jitter_config["hue"],
                p=color_jitter_config["probability"],
            )
        )

    if normalize_config := _get_augmentation_config(augmentation_config, "normalize"):
        transforms.append(A.Normalize(mean=normalize_config["mean"], std=normalize_config["std"]))

    transforms.append(ToTensorV2())

    return A.Compose(transforms, bbox_params=_bbox_params())
