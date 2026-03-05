#!/usr/bin/env python3
"""
All plotting functions for experiment analysis.

Cross-experiment (study directory):
    from analysis.load import load_results
    from analysis.plot import metric_vs_param, convergence_curves, compare_algorithms

    results = load_results("results/my_study")
    metric_vs_param(results, 'lmbda', 'PSNR', 'figs/')
    convergence_curves(results, group_by='lmbda', output_dir='figs/')

    ctv = load_results("results/ctv_study")
    ga  = load_results("results/gradalign_study")
    compare_algorithms({'CTV': ctv, 'GradAlign': ga}, 'PSNR', 'figs/')

Single experiment (zarr path):
    from analysis.plot import visualize
    visualize('results/exp/results.zarr', 'figs/', rgb_indices=[20, 10, 5])

CLI:
    python analysis/plot.py --zarr_path results/exp/results.zarr --output_dir figs/
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import zarr
from pathlib import Path

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .load import metric as get_metric


# ── helpers ──────────────────────────────────────────────────────────────────

def _save(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def _to_rgb(image, rgb_indices):
    """Select bands and normalize to [0,1] for display. image: [C, H, W]"""
    if rgb_indices is not None:
        bands = image[rgb_indices]
    else:
        bands = image[:3] if image.shape[0] >= 3 else np.repeat(image[:1], 3, axis=0)
    lo, hi = bands.min(), bands.max()
    return np.transpose((bands - lo) / (hi - lo + 1e-8), (1, 2, 0))


# ── cross-experiment plots ────────────────────────────────────────────────────

def metric_vs_param(results, param, metric_name, output_dir, title=None):
    """Plot mean metric grouped by param value (line plot or bar chart)."""
    groups = {}
    for r in results:
        groups.setdefault(r.get(param, 'unknown'), []).append(get_metric(r, metric_name))

    xs = sorted(groups.keys(), key=lambda x: (isinstance(x, str), x))
    ys = [np.mean(groups[x]) for x in xs]

    plt.figure(figsize=(8, 5))
    if xs and isinstance(xs[0], str):
        plt.bar(range(len(xs)), ys, alpha=0.8)
        plt.xticks(range(len(xs)), [str(x) for x in xs], rotation=45, ha='right')
    else:
        plt.plot(xs, ys, 'o-', linewidth=2, markersize=8)
    plt.xlabel(param)
    plt.ylabel(metric_name)
    plt.title(title or f'{metric_name} vs {param}')
    plt.grid(True, alpha=0.3)
    _save(f'{output_dir}/{metric_name}_vs_{param}.png')


def convergence_curves(results, group_by, output_dir, title=None):
    """Plot mean loss curves grouped by a parameter (semilogy)."""
    groups = {}
    for r in results:
        groups.setdefault(r.get(group_by, 'unknown'), []).append(r['loss'][0])

    plt.figure(figsize=(10, 6))
    for k, curves in groups.items():
        plt.semilogy(np.mean(curves, axis=0), label=f'{group_by}={k}', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title or f'Convergence by {group_by}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save(f'{output_dir}/convergence_by_{group_by}.png')


def compare_algorithms(results_by_alg, metric_name, output_dir, title=None):
    """Boxplot comparing algorithms on a metric."""
    data = {alg: [get_metric(r, metric_name) for r in res]
            for alg, res in results_by_alg.items()}

    plt.figure(figsize=(8, 5))
    plt.boxplot(list(data.values()), labels=list(data.keys()), patch_artist=True,
                boxprops=dict(facecolor='skyblue', alpha=0.7),
                medianprops=dict(color='red'))
    plt.ylabel(metric_name)
    plt.title(title or f'{metric_name} by Algorithm')
    plt.grid(True, alpha=0.3, axis='y')
    _save(f'{output_dir}/{metric_name}_comparison.png')


# ── single-experiment plots ───────────────────────────────────────────────────

def _simulate_inputs(results, dataset_path):
    """
    Re-simulate Y_H and Y_M for image 0 using stored experiment params.
    noise=False: shows degradation structure (blur+downsample, spectral avg) without
    a stochastic noise realization.
    """
    import torch
    from src.datasets.pandataset import PANDataset

    dataset = PANDataset(
        root_dir=dataset_path,
        split='train',
        normalize=True,
        scale=int(results.get('scale', 4)),
        sigma_blur=float(results.get('sigma_blur', 1.0)),
        noise_level=float(results.get('noise_level', 40)),
        device='cpu',
        seed=int(results.get('seed', 42))
    )
    X = dataset[0].unsqueeze(0)
    Y_H = dataset.simulate_low_res_hsi(X, noise=False).squeeze(0).cpu().numpy()
    Y_M = dataset.simulate_panchromatic(X, noise=False).squeeze(0).cpu().numpy()
    return Y_H, Y_M


def visualize(zarr_path, output_dir, rgb_indices=None, dataset_path=None):
    """
    Visualize a single experiment. Generates up to three figures:
      convergence.png     — loss curve (always)
      reconstruction.png  — original / reconstructed / error
      inputs.png          — LR HSI and panchromatic inputs

    dataset_path is optional: falls back to the path stored in the zarr attrs.
    """
    root = zarr.open(zarr_path, mode='r')
    results = {'reconstructed': root['reconstructed'][:], 'loss': root['loss'][:]}
    results.update(dict(root.attrs))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    algorithm = results.get('algorithm', 'Algorithm')

    # Convergence
    plt.figure(figsize=(8, 5))
    plt.semilogy(results['loss'][0])
    plt.xlabel('Iteration'); plt.ylabel('Loss')
    plt.title(f'{algorithm} Convergence'); plt.grid(True, alpha=0.3)
    _save(f'{output_dir}/convergence.png')

    dataset_path = dataset_path or results.get('dataset_path')
    if dataset_path is None:
        print('No dataset_path — skipping reconstruction and inputs plots.')
    else:
        # Reconstruction
        try:
            ds = zarr.open(dataset_path, mode='r')
            original     = np.transpose(ds['train/0'][:], (2, 0, 1))
            reconstructed = results['reconstructed'][0]
            orig_rgb  = _to_rgb(original, rgb_indices)
            recon_rgb = _to_rgb(reconstructed, rgb_indices)
            error = np.abs(orig_rgb - recon_rgb).mean(axis=2)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(orig_rgb);   axes[0].set_title('Original');      axes[0].axis('off')
            axes[1].imshow(recon_rgb);  axes[1].set_title('Reconstructed'); axes[1].axis('off')
            im = axes[2].imshow(error, cmap='viridis')
            axes[2].set_title('Error'); axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            plt.suptitle(algorithm)
            _save(f'{output_dir}/reconstruction.png')
        except Exception as e:
            print(f'Skipping reconstruction plot: {e}')

        # Inputs
        try:
            Y_H, Y_M = _simulate_inputs(results, dataset_path)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(_to_rgb(Y_H, rgb_indices))
            axes[0].set_title('LR HSI (noiseless)'); axes[0].axis('off')
            pan = Y_M[0]
            axes[1].imshow((pan - pan.min()) / (pan.max() - pan.min() + 1e-8), cmap='gray')
            axes[1].set_title('Panchromatic (noiseless)'); axes[1].axis('off')
            _save(f'{output_dir}/inputs.png')
        except Exception as e:
            print(f'Skipping inputs plot: {e}')

    print(f'\n=== {algorithm} ===')
    for name in ['PSNR', 'SSIM', 'SAM', 'RNMSE', 'CC']:
        if f'{name}_mean' in results:
            print(f'  {name}: {results[f"{name}_mean"]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize a single experiment result')
    parser.add_argument('--zarr_path', required=True)
    parser.add_argument('--output_dir', default='visualization_results')
    parser.add_argument('--rgb_indices', type=int, nargs='+', default=None)
    parser.add_argument('--dataset_path', default=None,
                        help='Path to dataset zarr (optional: stored path is used if omitted)')
    args = parser.parse_args()
    visualize(args.zarr_path, args.output_dir, args.rgb_indices, args.dataset_path)
