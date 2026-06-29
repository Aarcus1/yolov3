import argparse
import json
from typing import List

def load_coco_class_names(annotation_file: str):
    with open(annotation_file, "r") as f:
        coco = json.load(f)
    categories = coco["categories"]
    categories = sorted(categories, key=lambda x: x["id"])
    class_names = [c["name"] for c in categories]
    ids = [c["id"] for c in categories]
    return class_names, ids

def main():
    parser = argparse.ArgumentParser(description="Load COCO class names from annotation file")
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to COCO annotation JSON file (e.g., instances_val2017.json)"
    )
    args = parser.parse_args()

    class_names, ids = load_coco_class_names(args.annotations)
    print(class_names)
    print(ids)

if __name__ == "__main__":
    main()
