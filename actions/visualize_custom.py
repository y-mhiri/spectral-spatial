#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.append(os.path.dirname(__file__))
from loaders import *
from visualization_helpers import *

def get_rgb_indices(dataset_path):
    """Get RGB band indices for dataset"""
    # This is dataset-specific - you may need to adjust
    rgb_indices_map = {
        'harvard': [29, 19, 9],  # Approximate RGB bands for Harvard dataset
        'pavia': [55, 41, 12],   # Example for Pavia
        'default': [0, 1, 2]     # Fallback
    }
    
    dataset_name = os.path.basename(dataset_path).split('.')[0].lower()
    return rgb_indices_map.get(dataset_name, rgb_indices_map['default'])

def visualize_group(metadata, types, target_size, output_dir, img_indices=None):
    """Visualize results from a single group"""
    group_path = metadata['group_path']
    
    # Load arrays for this group
    available_arrays = []
    if 'reconstructed_shape' in metadata.get('zarr_metadata', {}):
        available_arrays.append('reconstructed')
    if 'loss_shape' in metadata.get('zarr_metadata', {}):
        available_arrays.append('loss')
    
    arrays_data = load_zarr_arrays(group_path, available_arrays)
    
    # Get dataset info for RGB indices
    dataset_path = metadata.get('parameters', {}).get('dataset_path', '')
    rgb_indices = get_rgb_indices(dataset_path)
    
    # Load ground truth if needed
    ground_truth_data = None
    if 'groundtruth' in types or 'comparison' in types:
        # We need to reconstruct ground truth from dataset
        # This requires dataset loading - simplified here
        pass
    
    # Process each image index
    if img_indices is None:
        if 'reconstructed' in arrays_data:
            img_indices = range(arrays_data['reconstructed'].shape[0])
        else:
            img_indices = [0]
    
    for img_idx in img_indices:
        images_to_process = {}
        
        # Extract different image types
        if 'reconstructed' in types and 'reconstructed' in arrays_data:
            recon = arrays_data['reconstructed'][img_idx]  # [bands, H, W]
            
            if 'rgb' in types:
                rgb_recon = extract_rgb_image(recon, rgb_indices)
                filename = generate_filename(metadata, 'rgb_reconstructed', img_idx)
                filepath = os.path.join(output_dir, filename)
                save_image(rgb_recon, filepath, target_size)
                print(f"Saved: {filename}")
            
            if 'eigenimage' in types:
                eigen_recon = extract_eigenimage_rgb(recon)
                filename = generate_filename(metadata, 'eigenimage_reconstructed', img_idx)
                filepath = os.path.join(output_dir, filename)
                save_image(eigen_recon, filepath, target_size)
                print(f"Saved: {filename}")
            
            if 'bands' in types:
                # Save first few bands as grayscale
                for band_idx in [0, 10, 20, 30]:
                    if band_idx < recon.shape[0]:
                        band_img = extract_band_image(recon, band_idx)
                        filename = generate_filename(metadata, f'band{band_idx}_reconstructed', img_idx)
                        filepath = os.path.join(output_dir, filename)
                        save_image(band_img, filepath, target_size)
                
                print(f"Saved bands for image {img_idx}")
        
        # Add ground truth and comparison logic here when needed
        # This would require loading original dataset

def visualize_experiment(experiment_dir, groups, types, layout, target_size, img_indices):
    """Visualize results from experiment"""
    print("="*60)
    print(" CUSTOM VISUALIZATION")
    print("="*60)
    
    # Load metadata
    all_metadata = load_experiment_metadata(experiment_dir)
    successful = filter_successful_runs(all_metadata)
    
    if not successful:
        print("No successful runs found")
        return False
    
    # Filter by groups if specified
    if groups:
        filtered_metadata = []
        for metadata in successful:
            group_name = os.path.basename(metadata['group_path'])
            if group_name.startswith('group_'):
                try:
                    group_num = int(group_name.split('_')[1])
                    if group_num in groups:
                        filtered_metadata.append(metadata)
                except ValueError:
                    continue
        successful = filtered_metadata
    
    if not successful:
        print(f"No matching groups found: {groups}")
        return False
    
    print(f"Processing {len(successful)} groups...")
    
    # Create output directory
    output_dir = os.path.join(experiment_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each group
    for i, metadata in enumerate(successful):
        print(f"\nProcessing group {i+1}/{len(successful)}")
        try:
            visualize_group(metadata, types, target_size, output_dir, img_indices)
        except Exception as e:
            print(f"Error processing group: {e}")
            continue
    
    print(f"\nVisualization complete. Results in: {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Custom visualization of experiment results')
    
    # Core arguments
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Experiment directory containing results')
    
    # Selection arguments
    parser.add_argument('--groups', type=int, nargs='+', 
                       help='Specific group numbers to visualize')
    parser.add_argument('--algorithm', type=str,
                       help='Filter by algorithm name')
    parser.add_argument('--best', type=int,
                       help='Show N best performing groups')
    
    # Visualization options
    parser.add_argument('--types', nargs='+', 
                       choices=['rgb', 'eigenimage', 'bands', 'reconstructed', 'groundtruth', 'comparison'],
                       default=['rgb', 'reconstructed'],
                       help='Types of visualizations to generate')
    
    parser.add_argument('--layout', choices=['separate', 'grid'], default='separate',
                       help='Layout for multiple images')
    
    parser.add_argument('--target_size', type=int,
                       help='Target image size (default: original size)')
    
    parser.add_argument('--img_indices', type=int, nargs='+',
                       help='Specific image indices to visualize')
    
    args = parser.parse_args()
    
    # Process selection
    groups = args.groups
    if args.best:
        # Load metadata and find best groups
        all_metadata = load_experiment_metadata(args.experiment_dir)
        if args.algorithm:
            all_metadata = filter_by_algorithm(all_metadata, args.algorithm)
        
        successful = filter_successful_runs(all_metadata)
        
        # Sort by PSNR and take top N
        groups_with_psnr = []
        for metadata in successful:
            psnr_vals = extract_metric_value(metadata, 'psnr')
            if psnr_vals:
                avg_psnr = np.mean(psnr_vals)
                group_name = os.path.basename(metadata['group_path'])
                if group_name.startswith('group_'):
                    try:
                        group_num = int(group_name.split('_')[1])
                        groups_with_psnr.append((group_num, avg_psnr))
                    except ValueError:
                        continue
        
        groups_with_psnr.sort(key=lambda x: x[1], reverse=True)
        groups = [g[0] for g in groups_with_psnr[:args.best]]
        print(f"Selected best {len(groups)} groups: {groups}")
    
    success = visualize_experiment(args.experiment_dir, groups, args.types, 
                                 args.layout, args.target_size, args.img_indices)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()