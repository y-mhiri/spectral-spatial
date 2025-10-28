# /bin/sh


# Show ground truth
python cli/show_dataset.py --storage_path plots_and_figures/threshold/ --dataset_path ~/data/harvard_small.zarr/ --image_idx '0 1 2 3' --plots ground_truth --save_fig

# Noise level 0dB 

## Show simulated images 
python cli/show_dataset.py --storage_path plots_and_figures/threshold/ --dataset_path ~/data/harvard_small.zarr/ --image_idx '0 1 2 3' --noise_level 50 --sigma_blur 2 --scale 4 --plots simulated_hsi simulated_pan --save_fig

## Show criterion histogram and mask image
python cli/show_dataset.py --storage_path plots_and_figures/threshold/ --dataset_path ~/data/harvard_small.zarr/ --image_idx '0 1 2 3' --noise_level 50 --sigma_blur 2 --scale 4 --tau 1.0 --plots criterion_histogram mask_image --save_fig
python cli/show_dataset.py --storage_path plots_and_figures/threshold/ --dataset_path ~/data/harvard_small.zarr/ --image_idx '0 1 2 3' --noise_level 50 --sigma_blur 2 --scale 4 --tau 1e-3 --plots criterion_histogram mask_image --save_fig
python cli/show_dataset.py --storage_path plots_and_figures/threshold/ --dataset_path ~/data/harvard_small.zarr/ --image_idx '0 1 2 3' --noise_level 50 --sigma_blur 2 --scale 4 --tau 1e-4 --plots criterion_histogram mask_image --save_fig
python cli/show_dataset.py --storage_path plots_and_figures/threshold/ --dataset_path ~/data/harvard_small.zarr/ --image_idx '0 1 2 3' --noise_level 50 --sigma_blur 2 --scale 4 --tau 1e-5 --plots criterion_histogram mask_image --save_fig
python cli/show_dataset.py --storage_path plots_and_figures/threshold/ --dataset_path ~/data/harvard_small.zarr/ --image_idx '0 1 2 3' --noise_level 50 --sigma_blur 2 --scale 4 --tau 2e-5 --plots criterion_histogram mask_image --save_fig
python cli/show_dataset.py --storage_path plots_and_figures/threshold/ --dataset_path ~/data/harvard_small.zarr/ --image_idx '0 1 2 3' --noise_level 50 --sigma_blur 2 --scale 4 --tau 4e-5 --plots criterion_histogram mask_image --save_fig
python cli/show_dataset.py --storage_path plots_and_figures/threshold/ --dataset_path ~/data/harvard_small.zarr/ --image_idx '0 1 2 3' --noise_level 50 --sigma_blur 2 --scale 4 --tau 6e-5 --plots criterion_histogram mask_image --save_fig
python cli/show_dataset.py --storage_path plots_and_figures/threshold/ --dataset_path ~/data/harvard_small.zarr/ --image_idx '0 1 2 3' --noise_level 50 --sigma_blur 2 --scale 4 --tau 8e-5 --plots criterion_histogram mask_image --save_fig
python cli/show_dataset.py --storage_path plots_and_figures/threshold/ --dataset_path ~/data/harvard_small.zarr/ --image_idx '0 1 2 3' --noise_level 50 --sigma_blur 2 --scale 4 --tau 1e-6 --plots criterion_histogram mask_image --save_fig


