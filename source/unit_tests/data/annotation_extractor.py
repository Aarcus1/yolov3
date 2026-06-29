import json
import os
from pycocotools.coco import COCO

def reduce_annotations_to_subset(annotation_file, images_folder, output_file):
    coco = COCO(annotation_file)
    # Get all image filenames in the subset folder
    image_files = set(os.listdir(images_folder))
    # Find image IDs in COCO whose file_name is in the subset folder
    subset_image_ids = [img_id for img_id, img in coco.imgs.items() if img['file_name'] in image_files]
    # Fix category_id type in annotations to int
    ann_ids = coco.getAnnIds(imgIds=subset_image_ids)
    fixed_annotations = []
    for a in ann_ids:
        ann = coco.anns[a].copy()
        if 'category_id' in ann:
            ann['category_id'] = int(ann['category_id'])
        fixed_annotations.append(ann)
    # Build reduced annotation dict
    reduced = {
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', []),
        'images': [coco.imgs[i] for i in subset_image_ids],
        'annotations': fixed_annotations,
        'categories': coco.dataset.get('categories', [])
    }
    with open(output_file, 'w') as f:
        json.dump(reduced, f)
    print(f"Reduced annotation file written to {output_file} (images: {len(reduced['images'])}, annotations: {len(reduced['annotations'])})")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reduce COCO annotation file to subset in given images folder')
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to original COCO annotation JSON')
    parser.add_argument('--images_folder', type=str, required=True, help='Path to folder containing subset images')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output reduced annotation JSON')
    args = parser.parse_args()
    reduce_annotations_to_subset(args.annotation_file, args.images_folder, args.output_file)
