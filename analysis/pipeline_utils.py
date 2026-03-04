#!/usr/bin/env python3
"""
Shared utilities for analysis pipelines.
"""

import os
import numpy as np
from glob import glob
from typing import Dict, List

from analysis.data_io import load_experiment_data


def load_study_results(study_dir: str) -> List[Dict]:
    """Load all experiment results from a study directory."""
    zarr_files = sorted(glob(f"{study_dir}/run_*/results.zarr"))
    if not zarr_files:
        raise ValueError(f"No results found in {study_dir}")

    results = []
    for zarr_file in zarr_files:
        try:
            data = load_experiment_data(zarr_file)
            results.append(data)
            print(f"  ✓ Loaded: {os.path.basename(os.path.dirname(zarr_file))}")
        except Exception as e:
            print(f"  ✗ Failed: {zarr_file} - {e}")

    if not results:
        raise ValueError("No valid results loaded")
    return results


def extract_metric(result: Dict, metric_name: str) -> float:
    """
    Extract a scalar metric value from a result dict.

    Metrics may be stored as a list (one value per image) or as a scalar.
    Returns the first element of a list, or the scalar directly.
    """
    val = result.get('metrics', {}).get(metric_name, 0)
    if isinstance(val, list):
        return val[0] if val else 0.0
    return float(val)
