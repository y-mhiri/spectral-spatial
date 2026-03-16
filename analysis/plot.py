#!/usr/bin/env python3
"""
All plotting functions for experiment analysis.

Cross-experiment:
    from analysis.load import load_results, filter_results
    from analysis.plot import metric_vs_param, convergence_curves, compare_algorithms

    results = load_results("results/my_study")
    metric_vs_param(results, 'lmbda', 'PSNR', 'figs/')
    convergence_curves(results, group_by='lmbda', facet_by='algorithm', output_dir='figs/')

Single-experiment sanity check:
    from analysis.load import load_scene
    from analysis.plot import sanity_check

    scene = load_scene(result)
    sanity_check(result, scene, output_dir='figs/')

Spatial comparison (CTV vs GradAlign):
    from analysis.plot import diff_error_map, sam_map, spectral_profile

    diff_error_map(r_ctv['reconstructed'][0], r_ga['reconstructed'][0],
                   scene['gt'], scene['ym'], label_a='CTV', label_b='GradAlign')
    sam_map(r_ctv['reconstructed'][0], scene['gt'])
    spectral_profile((y, x), scene['gt'], CTV=r_ctv['reconstructed'][0],
                     GradAlign=r_ga['reconstructed'][0])

Set P.SAVE_FIGURES = False to display figures inline in a notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from .load import metric as get_metric, filter_results
except ImportError:
    from analysis.load import metric as get_metric, filter_results


# Set to False in a notebook to display figures inline instead of saving to disk.
SAVE_FIGURES = True


# ── helpers ──────────────────────────────────────────────────────────────────

def _save(path):
    plt.tight_layout()
    if SAVE_FIGURES:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {path}')
    else:
        plt.show()


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


# ── single-experiment sanity check ───────────────────────────────────────────

def sanity_check(result, scene, output_dir, image_idx=0, rgb_indices=None):
    """Convergence, noisy inputs, and reconstruction for one experiment.

    Args:
        result:    one entry from load_results()
        scene:     output of load_scene(result)
        output_dir: where to save figures (ignored when SAVE_FIGURES=False)
        image_idx: which image in the batch to display
        rgb_indices: band indices for RGB preview

    Example:
        scene = load_scene(result)
        sanity_check(result, scene, output_dir='figs/')
    """
    algorithm = result.get('algorithm', 'Algorithm')
    recon     = result['reconstructed'][image_idx]   # [C, H, W]

    # Convergence
    loss_c = _loss_curve(result['loss'],    image_idx)
    relv_c = _relval_curve(result['relval'], image_idx)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(loss_c); axes[0].set_title('Loss');               axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(relv_c); axes[1].set_title('Relative variation'); axes[1].grid(True, alpha=0.3)
    fig.suptitle(f'{algorithm} — Convergence')
    _save(f'{output_dir}/{algorithm}_convergence.png')

    # Inputs
    ym_2d = scene['ym'][0]
    ym_2d = (ym_2d - ym_2d.min()) / (ym_2d.max() - ym_2d.min() + 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(_to_rgb(scene['gt'], rgb_indices)); axes[0].set_title('GT');     axes[0].axis('off')
    axes[1].imshow(_to_rgb(scene['yh'], rgb_indices)); axes[1].set_title('LR HSI'); axes[1].axis('off')
    axes[2].imshow(ym_2d, cmap='gray');                axes[2].set_title('PAN');    axes[2].axis('off')
    fig.suptitle(f'{algorithm} — Inputs')
    _save(f'{output_dir}/{algorithm}_inputs.png')

    # Reconstruction
    gt_rgb    = _to_rgb(scene['gt'], rgb_indices)
    recon_rgb = _to_rgb(recon,       rgb_indices)
    error     = np.abs(gt_rgb - recon_rgb).mean(axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(gt_rgb);    axes[0].set_title('GT');            axes[0].axis('off')
    axes[1].imshow(recon_rgb); axes[1].set_title('Reconstructed'); axes[1].axis('off')
    im = axes[2].imshow(error, cmap='hot')
    axes[2].set_title('Error map'); axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(f'{algorithm} — Reconstruction')
    _save(f'{output_dir}/{algorithm}_reconstruction.png')

    for name in ['PSNR', 'SSIM', 'SAM', 'RNMSE', 'CC']:
        if f'{name}_mean' in result:
            print(f'  {name}: {result[f"{name}_mean"]:.4f}')


# ── spatial comparison ────────────────────────────────────────────────────────

def diff_error_map(recon_a, recon_b, gt, pan, label_a='A', label_b='B', output_dir='.'):
    """Side-by-side error maps and differential map with PAN gradient contours.

    Args:
        recon_a, recon_b: [C, H, W] reconstructions to compare
        gt:               [C, H, W] ground truth
        pan:              [1, H, W] or [H, W] panchromatic image
        label_a, label_b: display names for the two reconstructions

    Example:
        diff_error_map(r_ctv['reconstructed'][0], r_ga['reconstructed'][0],
                       scene['gt'], scene['ym'],
                       label_a='CTV', label_b='GradAlign', output_dir='figs/')
    """
    error_a = np.abs(recon_a - gt).mean(axis=0)   # [H, W]
    error_b = np.abs(recon_b - gt).mean(axis=0)
    diff    = error_a - error_b                    # positive where B is better

    pan_2d   = pan[0] if pan.ndim == 3 else pan
    gy, gx   = np.gradient(pan_2d)
    pan_grad = np.sqrt(gx**2 + gy**2)

    vmax = max(error_a.max(), error_b.max())
    lim  = np.abs(diff).max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, err, label in zip(axes[:2], [error_a, error_b], [label_a, label_b]):
        im = ax.imshow(err, cmap='hot', vmin=0, vmax=vmax)
        ax.set_title(f'|error| {label}'); ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    im = axes[2].imshow(diff, cmap='RdBu_r', vmin=-lim, vmax=lim)
    axes[2].contour(pan_grad, levels=4, colors='k', alpha=0.3, linewidths=0.5)
    axes[2].set_title(f'error({label_a}) − error({label_b})   [blue = {label_b} better]')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    _save(f'{output_dir}/diff_error_{label_a}_vs_{label_b}.png')


def sam_map(recon, gt, output_dir='.', title='SAM (degrees)'):
    """Per-pixel spectral angle map. Returns the SAM array [H, W] for further use.

    Example:
        sam_ctv = sam_map(r_ctv['reconstructed'][0], scene['gt'], title='CTV')
        sam_ga  = sam_map(r_ga['reconstructed'][0],  scene['gt'], title='GradAlign')
    """
    dot  = (recon * gt).sum(axis=0)
    norm = np.linalg.norm(recon, axis=0) * np.linalg.norm(gt, axis=0)
    sam  = np.degrees(np.arccos(np.clip(dot / (norm + 1e-8), -1, 1)))

    plt.figure(figsize=(7, 5))
    im = plt.imshow(sam, cmap='hot')
    plt.colorbar(im, fraction=0.046, label='degrees')
    plt.title(title); plt.axis('off')
    _save(f'{output_dir}/sam_map.png')
    return sam


def spectral_profile(pixel_yx, gt, output_dir='.', **reconstructions):
    """Spectral profiles at one pixel for GT and any number of reconstructions.

    Args:
        pixel_yx:          (y, x) pixel coordinates
        gt:                [C, H, W] ground truth
        **reconstructions: name=array pairs, each [C, H, W]

    Example:
        spectral_profile((120, 80), scene['gt'], output_dir='figs/',
                         CTV=r_ctv['reconstructed'][0],
                         GradAlign=r_ga['reconstructed'][0])
    """
    y, x = pixel_yx
    plt.figure(figsize=(10, 4))
    plt.plot(gt[:, y, x], 'k-', linewidth=2, label='GT')
    for name, recon in reconstructions.items():
        plt.plot(recon[:, y, x], '--', linewidth=1.5, label=name)
    plt.xlabel('Band index'); plt.ylabel('Intensity')
    plt.title(f'Spectral profile at pixel ({y}, {x})')
    plt.legend(); plt.grid(True, alpha=0.3)
    _save(f'{output_dir}/spectral_profile_{y}_{x}.png')


