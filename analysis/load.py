"""Load experiment results for analysis."""
import zarr
from glob import glob


def load_results(study_dir):
    """
    Load all */results.zarr from study_dir.
    Returns a list of dicts: arrays (reconstructed, loss) + all zarr attrs (params, metrics).

    Example:
        results = load_results("results/convergence_study")
        psnrs = [metric(r, 'PSNR') for r in results]
    """
    files = sorted(glob(f"{study_dir}/*/results.zarr"))
    if not files:
        raise ValueError(f"No results found in {study_dir}")
    results = []
    for f in files:
        root = zarr.open(f, mode='r')
        data = {'reconstructed': root['reconstructed'][:], 'loss': root['loss'][:]}
        data.update(dict(root.attrs))
        results.append(data)
    print(f"Loaded {len(results)} experiments from {study_dir}")
    return results


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
