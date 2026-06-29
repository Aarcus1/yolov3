import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from tqdm import tqdm


def compute_coco_mean_std(image_dir, annotation_file, batch_size=32, num_workers=4):
    dataset = CocoDetection(root=image_dir, annFile=annotation_file)

    def collate_fn(batch):
        batch_images, _ = zip(*batch)
        return list(batch_images)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    pixel_count = 0

    for images in tqdm(loader, desc="Processing COCO images"):
        for img in images:
            img = F.to_tensor(img)
            channel_sum += img.sum(dim=[1, 2])
            channel_squared_sum += (img ** 2).sum(dim=[1, 2])
            pixel_count += img.shape[1] * img.shape[2]

    mean = channel_sum / pixel_count
    std = (channel_squared_sum / pixel_count - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


def main():
    parser = argparse.ArgumentParser(description="Compute COCO dataset mean and std")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to COCO images")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to COCO annotations JSON")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    mean, std = compute_coco_mean_std(
        args.image_dir,
        args.annotation_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print("Computed Mean:", mean)
    print("Computed Std:", std)

if __name__ == "__main__":
    main()
