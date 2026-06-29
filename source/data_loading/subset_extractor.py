import argparse
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import os

def select_coco_subset(annotation_file, output_file, image_ids=None, category_ids=None):
    coco = COCO(annotation_file)
    subset = {
        "info": coco.dataset.get("info", {}),
        "licenses": coco.dataset.get("licenses", []),
        "images": [],
        "annotations": [],
        "categories": coco.dataset.get("categories", [])
    }

    # Select images
    if image_ids is None:
        image_ids = coco.getImgIds()
    if category_ids is not None:
        image_ids = coco.getImgIds(catIds=category_ids)

    subset["images"] = [coco.imgs[i] for i in image_ids]

    # Select annotations
    ann_ids = coco.getAnnIds(imgIds=image_ids, catIds=category_ids)
    subset["annotations"] = [coco.anns[i] for i in ann_ids]

    # Optionally filter categories
    if category_ids is not None:
        subset["categories"] = [cat for cat in subset["categories"] if cat["id"] in category_ids]

    # Write subset to file
    with open(output_file, "w") as f:
        json.dump(subset, f)
    print(f"Subset written to {output_file} (images: {len(subset['images'])}, annotations: {len(subset['annotations'])})")


def main():
    parser = argparse.ArgumentParser(description="Select a subset from COCO dataset and write to file.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to original COCO annotation JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output subset JSON")
    parser.add_argument("--image_ids", type=str, default=None, help="Comma-separated list of image IDs to include")
    parser.add_argument("--category_ids", type=str, default=None, help="Comma-separated list of category IDs to include")
    args = parser.parse_args()

    image_ids = [int(i) for i in args.image_ids.split(",") if args.image_ids] if args.image_ids else None
    category_ids = [int(i) for i in args.category_ids.split(",") if args.category_ids] if args.category_ids else None

    select_coco_subset(args.annotation_file, args.output_file, image_ids=image_ids, category_ids=category_ids)

if __name__ == "__main__":
    main()

