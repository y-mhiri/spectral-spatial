#!/usr/bin/env python3
"""
All plotting functions for experiment analysis.

Cross-experiment (study directory):
    from analysis.load import load_results, filter_results
    from analysis.plot import metric_vs_param, convergence_curves, compare_algorithms

    results = load_results("results/my_study")
    metric_vs_param(results, 'lmbda', 'PSNR', 'figs/')

    # single-panel: lines by lambda
    convergence_curves(results, group_by='lmbda', output_dir='figs/')

    # faceted: one panel per algorithm, lines by CP iterations (fix lambda first)
    subset = filter_results(results, lmbda=0.001)
    convergence_curves(subset, group_by='max_iter_cp', facet_by='algorithm', output_dir='figs/')

    ctv = load_results("results/ctv_study")
    ga  = load_results("results/gradalign_study")
    compare_algorithms({'CTV': ctv, 'GradAlign': ga}, 'PSNR', 'figs/')

Single experiment (zarr path):
    from analysis.plot import visualize, visualize_gradalign
    visualize('results/exp/results.zarr',          'figs/', rgb_indices=[20, 10, 5])
    visualize_gradalign('results/exp/results.zarr', 'figs/', rgb_indices=[20, 10, 5])

CLI:
    python analysis/plot.py --zarr_path results/exp/results.zarr
    python analysis/plot.py --zarr_path results/exp/results.zarr --gradalign
    python analysis/plot.py --study_dir results/convergence_study_...   # all experiments
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

try:
    from .load import metric as get_metric, filter_results
except ImportError:
    from analysis.load import metric as get_metric, filter_results


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


# ── helpers ──────────────────────────────────────────────────────────────────

def _loss_curve(loss_array, idx=0):
    """Extract one loss curve from loss_array [N_images, max_iter], masking trailing zeros."""
    curve = loss_array[idx].astype(float)
    if curve[-1] == 0:
        last = np.flatnonzero(curve)
        if len(last):
            curve[last[-1] + 1:] = np.nan
    return curve


def _distance_curve(loss_array, idx=0):
    """loss - loss_final: distance to apparent optimum. Values <= 0 masked (avoids log(0))."""
    curve = _loss_curve(loss_array, idx)
    valid = curve[~np.isnan(curve)]
    if len(valid) == 0:
        return curve
    dist = curve - valid[-1]          # subtract last non-NaN value (where algorithm stopped)
    dist[dist <= 0] = np.nan
    return dist


def _relval_curve(relval_array, idx=0):
    """Relative variation ||U_{k+1} - U_k|| / ||U_k||, trailing zeros masked."""
    curve = relval_array[idx].astype(float)
    if curve[-1] == 0:
        last = np.flatnonzero(curve)
        if len(last):
            curve[last[-1] + 1:] = np.nan
    return curve


def _add_zoom_inset(ax, zoom_tail):
    """Add a top-right inset showing the last zoom_tail fraction of all lines on ax."""
    lines = [l for l in ax.get_lines() if len(l.get_xdata()) > 0]
    if not lines:
        return
    n = len(lines[0].get_ydata())
    start = int(n * (1 - zoom_tail))
    axins = ax.inset_axes([0.45, 0.45, 0.52, 0.52])
    for line in lines:
        x, y = line.get_xdata(), line.get_ydata()
        axins.semilogy(x[start:], y[start:],
                       color=line.get_color(), linewidth=line.get_linewidth())
    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=7)
    axins.set_title(f'last {int(zoom_tail * 100)}%', fontsize=8)


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


_CURVE_MODES = {
    'loss':     ('Loss',              lambda r: _loss_curve(r['loss'])),
    'distance': ('Loss \u2212 final', lambda r: _distance_curve(r['loss'])),
    'relval':   ('Relative variation',lambda r: _relval_curve(r['relval'])),
}


def convergence_curves(results, group_by, output_dir, facet_by=None, title=None,
                       mode='loss', burn_in=0, zoom_tail=None):
    """Plot convergence curves grouped by a parameter (semilog-y).

    Args:
        mode (str):        'loss' | 'distance' (loss−final) | 'relval' (relative variation).
        burn_in (int):     Skip the first burn_in iterations (common identical descent phase).
        zoom_tail (float): If set (e.g. 0.3), add an inset showing the last 30% of iterations.
        facet_by (str):    Optional second parameter creating one subplot per value.

    Examples:
        convergence_curves(results, group_by='lmbda', output_dir='figs/', mode='distance')
        convergence_curves(results, group_by='lmbda', output_dir='figs/', mode='relval', zoom_tail=0.3)
        subset = filter_results(results, lmbda=0.001)
        convergence_curves(subset, group_by='max_iter_cp', facet_by='algorithm',
                           output_dir='figs/', mode='distance', burn_in=10)
    """
    ylabel, load_curve = _CURVE_MODES[mode]

    def _plot_groups(ax, subset, group_by):
        groups = {}
        for r in subset:
            groups.setdefault(r.get(group_by, 'unknown'), []).append(load_curve(r))
        for k, curves in sorted(groups.items(), key=lambda x: (isinstance(x[0], str), x[0])):
            mean = np.nanmean(curves, axis=0)[burn_in:]
            iters = np.arange(burn_in, burn_in + len(mean))
            ax.semilogy(iters, mean, label=f'{group_by}={k}', linewidth=2)
        if zoom_tail is not None:
            _add_zoom_inset(ax, zoom_tail)

    prefix = '' if mode == 'loss' else f'{mode}_'

    if facet_by is None:
        plt.figure(figsize=(10, 6))
        _plot_groups(plt.gca(), results, group_by)
        plt.xlabel('Iteration')
        plt.ylabel(ylabel)
        plt.title(title or f'Convergence ({mode}) by {group_by}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        _save(f'{output_dir}/{prefix}convergence_by_{group_by}.png')
    else:
        facets = sorted(set(r.get(facet_by, 'unknown') for r in results),
                        key=lambda x: (isinstance(x, str), x))
        fig, axes = plt.subplots(1, len(facets), figsize=(7 * len(facets), 5), sharey=True)
        if len(facets) == 1:
            axes = [axes]

        for ax, facet_val in zip(axes, facets):
            subset = [r for r in results if r.get(facet_by) == facet_val]
            _plot_groups(ax, subset, group_by)
            ax.set_title(f'{facet_by}={facet_val}')
            ax.set_xlabel('Iteration')
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(ylabel)
        fig.suptitle(title or f'Convergence ({mode}) by {group_by}, faceted by {facet_by}')
        _save(f'{output_dir}/{prefix}convergence_{group_by}_by_{facet_by}.png')


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

def _load_dataset(results, dataset_path):
    """Instantiate PANDataset from stored experiment params."""
    import torch
    from src.datasets.pandataset import PANDataset
    dataset = PANDataset(
        root_dir=dataset_path, split='train', normalize=True,
        scale=int(results.get('scale', 4)),
        sigma_blur=float(results.get('sigma_blur', 1.0)),
        noise_level=float(results.get('noise_level', 40)),
        device='cpu', seed=int(results.get('seed', 42))
    )
    return dataset


def visualize(zarr_path, output_dir, rgb_indices=None, dataset_path=None):
    """
    Visualize a single experiment.

    Outputs (in output_dir):
      convergence.png     — loss curve
      inputs.png          — ground truth | LR HSI | panchromatic
      reconstruction.png  — ground truth | reconstructed | error map

    dataset_path is optional: falls back to the path stored in zarr attrs.
    """
    root = zarr.open(zarr_path, mode='r')
    results = dict(root.attrs)
    results['reconstructed'] = root['reconstructed'][:]
    results['loss']   = root['loss'][:]
    results['relval'] = root['relval'][:]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    algorithm = results.get('algorithm', 'Algorithm')

    # Convergence — 3 panels: loss | loss−final | relval (semilog-y, image 0)
    loss_c = _loss_curve(results['loss'])
    dist_c = _distance_curve(results['loss'])
    relv_c = _relval_curve(results['relval'])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, curve, label in zip(axes,
                                 [loss_c,   dist_c,           relv_c],
                                 ['Loss', 'Loss \u2212 final', 'Relative variation']):
        ax.semilogy(curve)
        ax.set_xlabel('Iteration'); ax.set_ylabel(label)
        ax.set_title(label); ax.grid(True, alpha=0.3)
    fig.suptitle(f'{algorithm} — Convergence')
    _save(f'{output_dir}/convergence.png')

    dataset_path = dataset_path or results.get('dataset_path')
    if dataset_path is None:
        print('No dataset_path — skipping inputs and reconstruction plots.')
        _print_metrics(results, algorithm)
        return

    dataset = _load_dataset(results, dataset_path)
    rgb_indices = dataset.rgb_index if rgb_indices is None else rgb_indices

    X = dataset[0].unsqueeze(0)
    original     = np.transpose(zarr.open(dataset_path, mode='r')['train/0'][:], (2, 0, 1))
    Y_H          = dataset.simulate_low_res_hsi(X, noise=True).squeeze(0).cpu().numpy()
    Y_M          = dataset.simulate_panchromatic(X, noise=True).squeeze(0).cpu().numpy()
    pan_norm     = (Y_M[0] - Y_M[0].min()) / (Y_M[0].max() - Y_M[0].min() + 1e-8)

    # Inputs: GT | LR HSI | PAN
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(_to_rgb(original, rgb_indices)); axes[0].set_title('Ground truth');       axes[0].axis('off')
    axes[1].imshow(_to_rgb(Y_H, rgb_indices));      axes[1].set_title('LR HSI (noisy)'); axes[1].axis('off')
    axes[2].imshow(pan_norm, cmap='gray');           axes[2].set_title('Panchromatic (noisy)');       axes[2].axis('off')
    plt.suptitle(algorithm)
    _save(f'{output_dir}/inputs.png')

    # Reconstruction: GT | Reconstructed | Error
    orig_rgb  = _to_rgb(original, rgb_indices)
    recon_rgb = _to_rgb(results['reconstructed'][0], rgb_indices)
    error     = np.abs(orig_rgb - recon_rgb).mean(axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_rgb);  axes[0].set_title('Ground truth');  axes[0].axis('off')
    axes[1].imshow(recon_rgb); axes[1].set_title('Reconstructed'); axes[1].axis('off')
    im = axes[2].imshow(error, cmap='hot')
    axes[2].set_title('Error map'); axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.suptitle(algorithm)
    _save(f'{output_dir}/reconstruction.png')

    _print_metrics(results, algorithm)


def _print_metrics(results, algorithm):
    print(f'\n=== {algorithm} ===')
    for name in ['PSNR', 'SSIM', 'SAM', 'RNMSE', 'CC']:
        if f'{name}_mean' in results:
            print(f'  {name}: {results[f"{name}_mean"]:.4f}')


def visualize_gradalign(zarr_path, output_dir, rgb_indices=None, dataset_path=None):
    """
    GradAlign-specific visualization. Calls visualize() then adds:
      gradalign_criterion.png  — criterion c(x,y) = ‖∇Y_M‖/Σ‖∇Y_M‖, with Otsu contour
      gradalign_mask.png       — alignment-active regions (c ≥ α) overlaid on PAN

    The criterion drives the Otsu threshold α that decides where hyperspectral
    gradients are aligned with the PAN structure.
    """
    import torch
    from src.algorithms.utils.nabla import nabla
    from src.algorithms.prox.tv_grad_align import compute_alpha_from_pan

    visualize(zarr_path, output_dir, rgb_indices, dataset_path)

    root = zarr.open(zarr_path, mode='r')
    results = dict(root.attrs)
    dataset_path = dataset_path or results.get('dataset_path')
    if dataset_path is None:
        print('No dataset_path — skipping GradAlign feature plots.')
        return

    dataset = _load_dataset(results, dataset_path)
    X   = dataset[0].unsqueeze(0)
    Y_M = dataset.simulate_panchromatic(X, noise=False)          # [1, 1, H, W]

    grad_panc = nabla(Y_M)                                        # [1, 1, H, W, 2]
    norm_grad = torch.norm(grad_panc.squeeze(), dim=-1)           # [H, W]
    c         = (norm_grad / (norm_grad.sum() + 1e-7)).cpu().numpy()
    alpha     = compute_alpha_from_pan(grad_panc)
    pan_norm  = (Y_M[0, 0].cpu().numpy())
    pan_norm  = (pan_norm - pan_norm.min()) / (pan_norm.max() - pan_norm.min() + 1e-8)

    # Criterion map with Otsu contour
    plt.figure(figsize=(8, 6))
    im = plt.imshow(c, cmap='viridis')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.contour(c, levels=[alpha], colors='red', linewidths=1)
    plt.title(f'GradAlign criterion c(x,y)   $\\alpha$ = {alpha:.2e} (Otsu)')
    plt.axis('off')
    _save(f'{output_dir}/gradalign_criterion.png')

    # Alignment mask overlaid on PAN
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(pan_norm, cmap='gray')
    axes[0].set_title('Panchromatic'); axes[0].axis('off')
    axes[1].imshow(pan_norm, cmap='gray')
    axes[1].imshow(c >= alpha, alpha=0.45, cmap='Reds')
    axes[1].set_title(f'Alignment mask  ($\\alpha$ = {alpha:.2e})'); axes[1].axis('off')
    plt.suptitle('GradAlign — edge alignment regions')
    _save(f'{output_dir}/gradalign_mask.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--zarr_path',  help='Path to a single results.zarr')
    group.add_argument('--study_dir',  help='Study directory (visualizes all experiments inside)')
    parser.add_argument('--output_dir',   default=None,
                        help='Output directory (default: next to zarr / <study_dir>/visualizations)')
    parser.add_argument('--rgb_indices',  type=int, nargs='+', default=None)
    parser.add_argument('--dataset_path', default=None,
                        help='Path to dataset zarr (optional: stored path is used if omitted)')
    parser.add_argument('--gradalign',    action='store_true',
                        help='Also produce GradAlign-specific feature figures')
    args = parser.parse_args()

    fn = visualize_gradalign if args.gradalign else visualize

    if args.zarr_path:
        out = args.output_dir or str(Path(args.zarr_path).parent / 'visualizations')
        fn(args.zarr_path, out, args.rgb_indices, args.dataset_path)
    else:
        from glob import glob
        zarr_paths = sorted(glob(f'{args.study_dir}/*/results.zarr'))
        if not zarr_paths:
            raise ValueError(f'No results.zarr found in {args.study_dir}')
        base_out = args.output_dir or f'{args.study_dir}/visualizations'
        for zp in zarr_paths:
            exp_name = Path(zp).parent.name
            fn(zp, f'{base_out}/{exp_name}', args.rgb_indices, args.dataset_path)
