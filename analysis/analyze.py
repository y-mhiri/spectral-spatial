#!/usr/bin/env python3
"""
Analyze results from a completed study.

Generates convergence curves, metric plots, and algorithm comparison figures
for all experiments in a study directory.

Usage:
    python analysis/analyze.py --study_dir results/convergence_study_... --group_by lmbda
    python analysis/analyze.py --study_dir results/robustness_study_...  --group_by noise_level
    python analysis/analyze.py --study_dir results/norm_study_...        --group_by p

Output is saved to <study_dir>/analysis/ by default.
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.load import load_results
from analysis.plot import metric_vs_param, convergence_curves, compare_algorithms

METRICS = ['PSNR', 'SSIM', 'SAM', 'RNMSE']


def main():
    parser = argparse.ArgumentParser(description='Analyze study results')
    parser.add_argument('--study_dir',  required=True,  help='Directory produced by a study_*.sh script')
    parser.add_argument('--group_by',   required=True,  help='Parameter to group by (e.g. lmbda, noise_level, p)')
    parser.add_argument('--output_dir', default=None,   help='Output directory (default: <study_dir>/analysis)')
    args = parser.parse_args()

    output_dir = args.output_dir or f'{args.study_dir}/analysis'

    results = load_results(args.study_dir)

    # Convergence curves and metrics vs swept parameter
    convergence_curves(results, group_by=args.group_by, output_dir=output_dir)
    for m in METRICS:
        metric_vs_param(results, args.group_by, m, output_dir)

    # CTV vs GradAlign comparison (all study scripts run both)
    ctv = [r for r in results if r.get('algorithm') == 'CTV']
    ga  = [r for r in results if r.get('algorithm') == 'GradAlign']
    if ctv and ga:
        for m in METRICS:
            compare_algorithms({'CTV': ctv, 'GradAlign': ga}, m, output_dir)
    else:
        print('Skipping algorithm comparison: results for only one algorithm found.')

    print(f'\nDone. Figures saved to: {output_dir}')


if __name__ == '__main__':
    main()
