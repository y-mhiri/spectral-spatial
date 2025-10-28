import argparse
import torch
import zarr
import os

import matplotlib.pyplot as plt

# Setup paths and imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.datasets.pandataset import PANDataset

from src.algorithms.nabla import nabla
from src.datasets.visualization import *
from skimage.filters import threshold_otsu
from torch.nn import Sigmoid

def main():
    parser = argparse.ArgumentParser()
    
    # Core parameters
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--storage_path", type=str, default='.')
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_fig", action='store_true')
    parser.add_argument("--plots", type=str, nargs='+')
    
    # Data parameterss
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_idx", type=str, default="0")
    parser.add_argument("--noise_level", type=float, default=0)
    parser.add_argument("--sigma_blur", type=float, default=4)
    parser.add_argument("--scale", type=int, default=4)
    
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tau", type=float, default=1.0)
    
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = args.device
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

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

    image_idx = [int(idx) for idx in args.image_idx.split(' ')]

    subset = torch.utils.data.Subset(dataset, image_idx)
    
    for j, data in enumerate(subset):
        print(f"Processing image {j+1}/{len(subset)}")
        
        # Prepare data
        X = data.unsqueeze(0)
        Y_M = dataset.simulate_panchromatic(X, noise=True).to(device=device, dtype=dtype)
        Y_H = dataset.simulate_low_res_hsi(X).to(device=device, dtype=dtype)        
        
        grad_panc = nabla(Y_M)
                
        grad_norm = torch.norm(grad_panc.squeeze(), dim=-1)  # Shape [H,W]
        criterion = grad_norm / (grad_norm.sum() + 1e-7)
        criterion_npy = criterion.cpu().numpy()

        
        threshold = threshold_otsu(criterion_npy) if args.threshold is None else args.threshold
        # threshold = torch.quantile(c_n, 0.75)

        print(f'threshold = {threshold:2g}')   


        mask = Sigmoid()((criterion - threshold)/args.tau)

        # Show ground truth image and simulated observations

        if 'ground_truth' in args.plots:
            X_rgb = extract_rgb_image(X.squeeze().numpy(), rgb_indices=dataset.rgb_index)
            plt.figure()
            plt.imshow(X_rgb)
            plt.axis('off')
            if args.save_fig:
                plt.savefig(os.path.join(args.storage_path, f'{j}_original_hsi_image.png'),bbox_inches='tight')
            
        if 'simulated_hsi' in args.plots:
            Yh_rgb = extract_rgb_image(Y_H.squeeze().numpy(), rgb_indices=dataset.rgb_index)
            plt.figure()
            plt.imshow(Yh_rgb)
            plt.axis('off')
            if args.save_fig:
                plt.savefig(os.path.join(args.storage_path, f'{j}_simulated_hsi.png'),bbox_inches='tight')

        if 'simulated_pan' in args.plots:
            plt.figure()
            plt.imshow(Y_M.squeeze())
            plt.axis('off')
            if args.save_fig:
                plt.savefig(os.path.join(args.storage_path, f'{j}_simulated_pan.png'),bbox_inches='tight')

        if 'mask_image' in args.plots:
            plt.figure()
            plt.imshow(mask, vmin=0, vmax=1, cmap='Spectral_r')
            plt.axis('off')
            # plt.title(f'Mask image $\\tau$ = {args.tau:2g}, threshold = {threshold:2g}')
            plt.colorbar()
            if args.save_fig:
                plt.savefig(os.path.join(args.storage_path, f'{j}_mask_image_tau_{args.tau:2g}_thresh_{threshold:2g}.png'),bbox_inches='tight')

        if 'criterion_histogram' in args.plots:
            plt.figure()
            plt.hist(criterion_npy.flatten(),bins=256)
            plt.axvline(x=threshold, color='red')
            if args.save_fig:
                plt.savefig(os.path.join(args.storage_path, f'{j}_criterion_hist_thresh={threshold:2g}.png'),bbox_inches='tight')
        
        if 'grad_norm_histogram' in args.plots:
            plt.figure()
            plt.hist(grad_norm.flatten(), bins=256)

            if args.save_fig:
                plt.savefig(os.path.join(args.storage_path, f'{j}_grad_norm_hist.png'),bbox_inches='tight')
        

    torch.cuda.empty_cache()
    if not args.save_fig: 
        plt.show()
    

if __name__ == "__main__":
    main()