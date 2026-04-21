"""Load experiment results for analysis."""
import zarr
from glob import glob
import numpy as np


def load_results(study_dir):
    """
    Load all */results.zarr from study_dir.
    Returns a list of dicts: arrays (reconstructed, loss) + all zarr attrs (params, metrics).

    Example:
        results = load_results("results/convergence_study")
        psnrs = [metric(r, 'PSNR') for r in results]
    """
    files = sorted(glob(f"{study_dir}/**/results.zarr", recursive=True))
    if not files:
        raise ValueError(f"No results found in {study_dir}")
    results = []
    for f in files:
        root = zarr.open(f, mode='r')
        data = {'zarr_path': f, 'reconstructed': root['reconstructed'][:], 'loss': root['loss'][:], 'relval': root['relval'][:]}
        data.update(dict(root.attrs))
        results.append(data)
    print(f"Loaded {len(results)} experiments from {study_dir}")
    return results


def filter_results(results, **conditions):
    """Select experiments matching exact parameter values.

    Example:
        subset = filter_results(results, lmbda=0.001, max_iter_cp=50)
    """
    return [r for r in results if all(r.get(k) == v for k, v in conditions.items())]


def to_dataframe(results):
    """Convert load_results() output to a pandas DataFrame, dropping large arrays.

    Example:
        results = load_results("results/aggregate")
        df = to_dataframe(results)
        df[['algorithm', 'lmbda', 'PSNR_mean']].sort_values('PSNR_mean', ascending=False)
    """
    import pandas as pd
    ARRAY_KEYS = {'reconstructed', 'loss', 'relval', 'zarr_path'}
    rows = [{k: v for k, v in r.items() if k not in ARRAY_KEYS} for r in results]
    return pd.DataFrame(rows)


def load_scene(result, image_idx=0):
    """Load GT, noisy LR HSI, and noisy PAN for one image from the stored dataset.

    Args:
        result:    one entry from load_results() — needs 'dataset_path' and degradation params.
        image_idx: which training image to load (default 0).

    Returns dict with numpy arrays:
        gt  [C, H, W]  — ground truth hyperspectral image
        yh  [C, h, w]  — noisy low-resolution HSI  (h = H // scale)
        ym  [1, H, W]  — noisy panchromatic image

    Example:
        scene = load_scene(result)
        sanity_check(result, scene, output_dir='figs/')
    """
    import torch
    import zarr as _zarr
    from src.datasets.pandataset import PANDataset

    dataset_path = result['dataset_path']
    dataset = PANDataset(
        root_dir=dataset_path, split='train', normalize=True,
        scale=int(result.get('scale', 4)),
        sigma_blur=float(result.get('sigma_blur', 1.0)),
        noise_level=float(result.get('noise_level', 40)),
        device='cpu', seed=int(result.get('seed', 42))
    )
    gt  = dataset[image_idx]
    X = gt.unsqueeze(0)
    # gt = np.transpose(_zarr.open(dataset_path, mode='r')[f'train/{image_idx}'][:], (2, 0, 1))
    yh = dataset.simulate_low_res_hsi(X, noise=True).squeeze(0).numpy()
    ym = dataset.simulate_panchromatic(X, noise=True).squeeze(0).numpy()
    return {'gt': gt.numpy(), 'yh': yh, 'ym': ym}, dataset.rgb_index


def metric(result, name):
    """
    Extract a scalar metric value from a result dict.
    Metrics are stored as {name}_mean by experiment_helpers.store_results().

    Example:
        psnr = metric(result, 'PSNR')
    """
    if f'{name}_mean' in result:
        return float(result[f'{name}_mean'])
    v = result.get(name, 0)
    return float(v[0]) if isinstance(v, list) else float(v)
