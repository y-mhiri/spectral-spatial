import argparse
import os
import zarr
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Create a cropped copy of harvard.zarr.")

    # Core parameters
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input harvard.zarr file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the cropped Zarr file.")
    parser.add_argument("--split", type=str, default="train", help="Split to process (train/test/val).")
    parser.add_argument("--image_indices", type=str, required=True, help="Indices of images to crop (e.g., '0 1 2').")
    parser.add_argument("--crop_centers", type=str, required=True, help="Crop centers (e.g., '256,256 512,512').")
    parser.add_argument("--crop_size", type=int, default=128, help="Size of the square crop.")

    args = parser.parse_args()

    # Load input Zarr
    input_zarr = zarr.open(args.input_path, mode='r')
    image_indices = list(map(int, args.image_indices.split()))
    crop_centers = [tuple(map(int, center.split(','))) for center in args.crop_centers.split()]

    # Create output Zarr
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output_zarr = zarr.open(args.output_path, mode='w')

    # Copy metadata
    for key, value in input_zarr.attrs.items():
        output_zarr.attrs[key] = value

    # Create output group for the split
    output_group = output_zarr.create_group(args.split)

    # Crop and save images
    for i, (idx, (y_center, x_center)) in enumerate(zip(image_indices, crop_centers)):
        img = input_zarr[args.split][str(idx)][:]
        height, width, channels = img.shape
        half_size = args.crop_size // 2

        # Calculate crop bounds
        y_start = max(0, y_center - half_size)
        y_end = min(height, y_center + half_size)
        x_start = max(0, x_center - half_size)
        x_end = min(width, x_center + half_size)

        # Crop and save
        cropped = img[y_start:y_end, x_start:x_end, :]
        output_group.create_array(name=str(i), data=cropped)

    print(f"Cropped dataset saved to {args.output_path}")

if __name__ == "__main__":
    main()
