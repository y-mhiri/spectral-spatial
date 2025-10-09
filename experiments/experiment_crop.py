import argparse
import os
import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Save RGB images and optionally crop them.")

    # Core parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the Zarr dataset.")
    parser.add_argument("--storage_path", type=str, required=True, help="Directory to save images.")
    parser.add_argument("--image_idx", type=int, required=True, nargs='+', help="Index of the image to process.")
    parser.add_argument("--crop_center", type=str, default=None, help="Center coordinates for cropping (y,x).")
    parser.add_argument("--crop_size", type=int, default=128, help="Size of the square crop.")

    args = parser.parse_args()

    # Load dataset
    dataset = zarr.open(args.dataset_path, mode='r')
    for idx in args.image_idx:
        img = torch.from_numpy(dataset['train'][str(idx)][:]).float()
        img = img.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)

        # Normalize for visualization
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Use RGB indices if available, else first 3 bands
        rgb_indices = dataset.attrs.get('rgb', [0, 1, 2])
        rgb_img = img[rgb_indices, :, :].permute(1, 2, 0).numpy()

        # Save RGB image
        os.makedirs(args.storage_path, exist_ok=True)
        plt.imsave(os.path.join(args.storage_path, f"image_{idx}_rgb.png"), rgb_img)

        # Crop if center is specified
        if args.crop_center:
            y_center, x_center = map(int, args.crop_center.split(','))
            half_size = args.crop_size // 2
            x_start = max(0, x_center - half_size)
            x_end = min(img.shape[2], x_center + half_size)
            y_start = max(0, y_center - half_size)
            y_end = min(img.shape[1], y_center + half_size)

            cropped = img[:, y_start:y_end, x_start:x_end]
            cropped_rgb = cropped[rgb_indices, :, :].permute(1, 2, 0).numpy()
            plt.imsave(os.path.join(args.storage_path, f"image_{idx}_cropped.png"), cropped_rgb)
            print(f"Cropped image saved to {args.storage_path}/image_{idx}_cropped.png")

if __name__ == "__main__":
    main()
