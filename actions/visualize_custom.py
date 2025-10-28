#!/usr/bin/env python3
import argparse
import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from loaders import *
from visualization_helpers import *

setup_paths()
from src.datasets.pandataset import PANDataset

def add_sub_figure(path, width):

    content = f"\\begin{{subfigure}}[b]{{{width:0.02f}}}\n"

    content += f"\includegraphics[width=\textwidth]{{{path}}}\n"
    content += "\end{subfigure}\n"

    return content

def generate_figure(path_to_pngs, nfigs_per_row=3, caption=None):

    nfig = len(arrays)
    
    content = ""
    content += "\\begin{figure}\n"

    for path in path_to_pngs:
        content += add_subfigure(path, width=1/nfigs_per_row)
    
    if caption is not None:
        content += caption

    content += "\end{figure}\n"

    return content


def get_rgb_indices(dataset_path):
    """Get RGB band indices for dataset"""
    rgb_indices_map = {
        'harvard': [29, 19, 9],
        'harvard_small': [29, 19, 9],
        'pavia': [55, 41, 12],
        'default': [0, 1, 2]
    }
    dataset_name = os.path.basename(dataset_path).split('.')[0].lower()
    return rgb_indices_map.get(dataset_name, rgb_indices_map['default'])

def load_all_image_data(metadata):
    """Load all image data (ground truth, reconstructed, noisy) for a group"""
    
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(**metadata.get('parameters', {}))
    

        # Setup experiment
    device = args.device
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(args.seed)

    # Open dataset
    dataset = PANDataset(
        root_dir=args.dataset_path,
        split='train',
        normalize=True,
        scale=args.scale,
        sigma_blur=args.sigma_blur,
        noise_level=args.noise_level,
        device=device,
        seed=args.seed
    )
    
    image_idx = getattr(args, 'image_idx', "6")
    
    image_idx = [int(idx) for idx in image_idx.split(' ')]
    subset = torch.utils.data.Subset(dataset, image_idx)
    
    # Load reconstructed arrays from zarr
    group_path = metadata['group_path']
    reconstructed_data = None
    try:
        arrays_data = load_zarr_arrays(group_path, ['reconstructed'])
        if 'reconstructed' in arrays_data:
            reconstructed_data = arrays_data['reconstructed']
    except Exception as e:
        print(f"Could not load reconstructed data: {e}")
    
    # Generate ground truth and noisy images
    all_data = {
        'groundtruth': [],
        'hsi_noisy': [],
        'pan_noisy': [],
        'reconstructed': []
    }
    
    for idx in range(len(subset)):
        gt_tensor = subset[idx]
        gt_batch = gt_tensor.unsqueeze(0)
        
        # Ground truth
        all_data['groundtruth'].append(gt_tensor.cpu().numpy())
        
        # Noisy HSI (low resolution)
        hsi_noisy = dataset.simulate_low_res_hsi(gt_batch)[0].cpu().numpy()
        all_data['hsi_noisy'].append(hsi_noisy)
        
        # Noisy panchromatic
        pan_noisy = dataset.simulate_panchromatic(gt_batch, noise=True)[0].cpu().numpy()
        all_data['pan_noisy'].append(pan_noisy)
        
        # Reconstructed (if available)
        if reconstructed_data is not None and idx < reconstructed_data.shape[0]:
            all_data['reconstructed'].append(reconstructed_data[idx])
    
    return all_data, getattr(args, 'algorithm', 'Unknown')

def save_visualization_set(image_data, group_num, img_idx, algorithm, rgb_indices, output_dir):
    """Save complete visualization set for one image"""
    
    viz_types = ['rgb']
    image_sources = ['groundtruth', 'hsi_noisy', 'reconstructed']
    
    for viz_type in viz_types:
        for source in image_sources:
            if source in image_data and len(image_data[source]) > img_idx:
                data = image_data[source][img_idx]
                
                # Extract visualization
                if viz_type == 'rgb':
                    img = extract_rgb_image(data, rgb_indices)
                    print(f'{source}')
                    print(f'mean : {np.mean(data)}')
                    print(f'std: {np.std(data)}')
                    print(f'max: {np.max(data)}')
                    print(f'min: {np.min(data)}')
                elif viz_type == 'eigenimage':
                    img = extract_eigenimage_rgb(data)
                
                # Generate filename
                filename = f"{group_num:03d}_{img_idx}_{algorithm}_{viz_type}_{source}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Save image
                save_image(img, filepath)
    
    # Save panchromatic if available
    if 'pan_noisy' in image_data and len(image_data['pan_noisy']) > img_idx:
        pan_data = image_data['pan_noisy'][img_idx]
        
        # Handle 3D panchromatic (squeeze if needed)
        if pan_data.ndim == 3 and pan_data.shape[0] == 1:
            pan_data = pan_data[0]
        
        # Normalize and save
        normalized_pan = normalize_image(pan_data)
        filename = f"{group_num:03d}_{img_idx}_{algorithm}_pan_noisy.png"
        filepath = os.path.join(output_dir, filename)
        save_image(normalized_pan, filepath)

def visualize_group(metadata, output_dir):
    """Generate all visualizations for one group"""
    # Extract group number from path
    group_name = os.path.basename(metadata['group_path'])
    group_num = int(group_name.split('_')[1]) if group_name.startswith('group_') else 0
    
    print(f"Processing group {group_num}...")
    
    try:
        # Load all image data
        image_data, algorithm = load_all_image_data(metadata)
        
        # Get dataset path for RGB indices
        dataset_path = extract_parameter_value(metadata, 'dataset_path', '')
        rgb_indices = get_rgb_indices(dataset_path)
        # Determine number of images to process
        num_images = len(image_data.get('groundtruth', []))
        
        # Generate visualizations for each image
        for img_idx in range(num_images):
            print(f'image {img_idx}')

            save_visualization_set(image_data, group_num, img_idx, algorithm, 
                                 rgb_indices, output_dir)
        
        print(f"  Saved {num_images} image sets")
        
    except Exception as e:
        print(f"  Error processing group {group_num}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate visualization images for experiment')
    parser.add_argument('--storage_path', type=str, required=True)                       
    parser.add_argument('--groups', type=int, nargs='+',
                       help='Specific group numbers to visualize (default: all successful)')
    parser.add_argument('--algorithm', type=str,
                       help='Filter by algorithm name')
    


    args = parser.parse_args()

    
    print("="*60)
    print(" VISUALIZATION GENERATOR")
    print("="*60)
    
    # Load metadata
    successful = load_experiment_metadata(args.storage_path)
    
    if not successful:
        print("No successful runs found")
        return
    
    # Filter by algorithm if specified
    if args.algorithm:
        successful = filter_by_algorithm(successful, args.algorithm)
        print(f"Filtered to {args.algorithm} algorithm")
    
    # Filter by groups if specified
    if args.groups:
        filtered = []
        for metadata in successful:
            group_name = os.path.basename(metadata['group_path'])
            if group_name.startswith('group_'):
                try:
                    group_num = int(group_name.split('_')[1])
                    if group_num in args.groups:
                        filtered.append(metadata)
                except ValueError:
                    continue
        successful = filtered
        print(f"Processing groups: {args.groups}")
    
    if not successful:
        print("No matching groups found")
        return
    
    print(f"Processing {len(successful)} groups...")
    
    # Create output directory
    output_dir = os.path.join(args.storage_path, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each group
    for metadata in successful:
        visualize_group(metadata, output_dir)

#     # Generate tex file

#     texfile_name = os.path.join(output_dir, 'figure.tex')

#     with open(texfile_name, 'w') as f:

#         # per image
#         # gt, noisy, pan, l111 x2, l221 x2, linf11 x2
        

#         f.write(generate_figure(path_to_pngs))
    
# # 

    
    print(f"\nVisualization complete!")
    print(f"Results saved to: {output_dir}")
    print(f"\nGenerated images:")
    print("- RGB and eigenimage versions of ground truth, reconstructed, and noisy HSI")
    print("- Panchromatic noisy images")
    print("- Files named: {group}_{img}_{algorithm}_{viz_type}_{source}.png")

if __name__ == "__main__":
    main()
