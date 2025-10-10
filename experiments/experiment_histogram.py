import argparse
import torch
import zarr
import os

import matplotlib.pyplot as plt

# Setup paths and imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from experiment_helpers import *

setup_paths()
from pansharpening import PANDataset
from pan_ctv import PANTVCB
from pan_ctv_grad_align import PANTVGradAlignment
from nabla import nabla
from metrics import compute_metrics
from skimage.filters import threshold_otsu
from torch.nn import Sigmoid

def main():
    parser = argparse.ArgumentParser()
    
    # Core parameters
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cuda_avail", type=bool, default=True)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    
    # Data parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_idx", type=str, default="6")
    parser.add_argument("--crop_center", type=bool, default=True)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--noise_level", type=float, required=True)
    parser.add_argument("--sigma", type=float, default=500)
    parser.add_argument("--scale", type=int, default=4)
    
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tau", type=float, nargs='+', default=[1.0])
    
    # PANTVGradAlign specific
    parser.add_argument("--threshold_type", type=str, default="soft", choices=["hard", "soft"])
    parser.add_argument("--threshold_param", type=float, default=0.5)
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    # Setup
    device, dtype = setup_device_and_dtype(args)
    dataset = create_dataset(args, device, dtype)

    image_idx = [int(idx) for idx in args.image_idx.split(' ')]

    subset = torch.utils.data.Subset(dataset, image_idx)
    
    # Run experiments
    for j, data in enumerate(subset):
        print(f"Processing image {j+1}/{len(subset)}")
        
        # Prepare data
        Y_M = dataset.get_panchromatic(data.unsqueeze(0), noise=True).to(device=device, dtype=dtype)
        
        grad_panc = nabla(Y_M)
                
        grad_norm = torch.norm(grad_panc.squeeze(), dim=-1)  # Shape [H,W]
        criterion = grad_norm / (grad_norm.sum() + 1e-7)
        c_n = criterion.cpu().numpy()

        
        alpha = threshold_otsu(c_n)
        # alpha = torch.quantile(c_n, 0.75)
        print(f'alpha = {alpha}')   

        for t in args.tau:
            mask = Sigmoid()((criterion - alpha)/t)

            plt.figure()
            # plt.imshow(Y_M.squeeze())
            plt.imshow(mask, vmin=0, vmax=1)
            plt.colorbar()
        # plt.savefig(os.path.join(args.storage_path, f'{j}_images.png'))

        plt.figure()
        plt.hist(c_n.flatten(),bins=256)
        plt.axvline(x=alpha, color='red')
        # plt.savefig(os.path.join(args.storage_path, f'{j}_criterion.png'))
        
        plt.figure()
        plt.hist(grad_norm.flatten(), bins=256)
        # plt.savefig(os.path.join(args.storage_path, f'{j}_grad_norm.png'))
    

    torch.cuda.empty_cache()
    
    plt.show()
    

if __name__ == "__main__":
    main()