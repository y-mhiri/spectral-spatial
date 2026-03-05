#!/usr/bin/env python3
"""
Convergence study analysis. Pairs with scripts/study_convergence.sh.

Answers three questions:
  1. How does the number of Chambolle-Pock sub-iterations affect convergence?
  2. How does the regularization parameter λ affect convergence?
  3. Do CTV and GradAlign behave the same under these variations?

Each question produces one figure: curves faceted by algorithm, averaged over the
other swept variable to show marginal effects.

Usage:
    python analysis/analyze_convergence.py --study_dir results/convergence_study_...
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.load import load_results, filter_results
from analysis.plot import convergence_curves, compare_algorithms

METRICS = ['PSNR', 'SSIM', 'SAM', 'RNMSE']


def main():
    parser = argparse.ArgumentParser(description='Analyze convergence study results')
    parser.add_argument('--study_dir',   required=True, help='Directory produced by scripts/study_convergence.sh')
    parser.add_argument('--output_dir',  default=None,  help='Output directory (default: <study_dir>/analysis)')
    parser.add_argument('--lmbda',       type=float,    default=None, help='Fix λ to a specific value')
    parser.add_argument('--max_iter_cp', type=int,      default=None, help='Fix CP iterations to a specific value')
    args = parser.parse_args()

    output_dir = args.output_dir or f'{args.study_dir}/analysis'
    results = load_results(args.study_dir)

    filters = {k: v for k, v in [('lmbda', args.lmbda), ('max_iter_cp', args.max_iter_cp)] if v is not None}
    if filters:
        results = filter_results(results, **filters)

    # Q1: Does CP iteration count affect convergence?
    # Lines = CP values, averaged over λ → marginal effect of inner solver.
    convergence_curves(results, group_by='max_iter_cp', facet_by='algorithm',
                       output_dir=output_dir, title='CP iterations impact')

    # Q2: Does λ affect convergence?
    # Lines = λ values, averaged over CP → marginal effect of regularization.
    convergence_curves(results, group_by='lmbda', facet_by='algorithm',
                       output_dir=output_dir, title='Regularization impact')

    # Q3: Which algorithm performs better across all conditions?
    ctv = [r for r in results if r.get('algorithm') == 'CTV']
    ga  = [r for r in results if r.get('algorithm') == 'GradAlign']
    if ctv and ga:
        for m in METRICS:
            compare_algorithms({'CTV': ctv, 'GradAlign': ga}, m, output_dir)

    print(f'\nDone. Figures saved to: {output_dir}')


if __name__ == '__main__':
    main()
