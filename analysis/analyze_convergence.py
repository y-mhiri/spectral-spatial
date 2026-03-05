#!/usr/bin/env python3
"""
Convergence study analysis. Pairs with scripts/study_convergence.sh.

Answers three questions:
  1. How does the number of Chambolle-Pock sub-iterations affect convergence?
  2. How does the regularization parameter λ affect convergence?
  3. Do CTV and GradAlign behave the same under these variations?

Each question is answered by fixing the other variable(s) and plotting convergence
curves faceted by algorithm — one figure per fixed value.

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
    parser.add_argument('--study_dir',  required=True, help='Directory produced by scripts/study_convergence.sh')
    parser.add_argument('--output_dir', default=None,  help='Output directory (default: <study_dir>/analysis)')
    args = parser.parse_args()

    output_dir = args.output_dir or f'{args.study_dir}/analysis'
    results = load_results(args.study_dir)

    # Infer swept values from loaded results
    lambdas  = sorted(set(r['lmbda']       for r in results if 'lmbda'       in r))
    cp_iters = sorted(set(r['max_iter_cp'] for r in results if 'max_iter_cp' in r))

    # ── Q1: CP sub-iterations impact ─────────────────────────────────────────
    # Fix λ, facet by algorithm → lines = CP iteration counts.
    # Shows whether more inner iterations accelerates or stabilizes convergence.
    print('\n--- Q1: CP sub-iterations impact (fixed λ) ---')
    for lmbda in lambdas:
        subset = filter_results(results, lmbda=lmbda)
        convergence_curves(
            subset,
            group_by='max_iter_cp',
            facet_by='algorithm',
            output_dir=f'{output_dir}/cp_impact',
            title=f'CP iterations impact  (λ={lmbda})',
        )

    # ── Q2: Regularization impact ─────────────────────────────────────────────
    # Fix CP iterations, facet by algorithm → lines = λ values.
    # Shows how strongly regularization affects convergence speed and stability.
    print('\n--- Q2: Regularization impact (fixed CP iterations) ---')
    for cp in cp_iters:
        subset = filter_results(results, max_iter_cp=cp)
        convergence_curves(
            subset,
            group_by='lmbda',
            facet_by='algorithm',
            output_dir=f'{output_dir}/lambda_impact',
            title=f'Regularization impact  (CP={cp} iters)',
        )

    # ── Q3: Algorithm comparison ──────────────────────────────────────────────
    # Fix both variables → direct CTV vs GradAlign comparison per condition.
    print('\n--- Q3: Algorithm comparison (fixed λ and CP iterations) ---')
    for lmbda in lambdas:
        for cp in cp_iters:
            subset = filter_results(results, lmbda=lmbda, max_iter_cp=cp)
            ctv = [r for r in subset if r.get('algorithm') == 'CTV']
            ga  = [r for r in subset if r.get('algorithm') == 'GradAlign']
            if ctv and ga:
                for m in METRICS:
                    compare_algorithms(
                        {'CTV': ctv, 'GradAlign': ga}, m,
                        output_dir=f'{output_dir}/algo_comparison/lambda_{lmbda}_cp_{cp}',
                    )

    print(f'\nDone. Figures saved to: {output_dir}')


if __name__ == '__main__':
    main()
