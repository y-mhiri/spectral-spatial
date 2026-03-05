#!/usr/bin/env python3
"""
Metric summary tables for experiment studies.

Core primitive — one function covers all study types:

    from analysis.load import load_results
    from analysis.tables import summarize, QUALITY_METRICS, CONVERGENCE_METRICS

    results = load_results('results/my_study')

    # Algorithm comparison (mean ± std across all conditions)
    summarize(results, group_by='algorithm', metrics=QUALITY_METRICS)

    # Convergence study: quality vs regularization
    summarize(results, group_by='lmbda',       metrics=QUALITY_METRICS)

    # Convergence study: inner solver impact
    summarize(results, group_by='max_iter_cp', metrics=CONVERGENCE_METRICS)

    # Monte Carlo: statistical reliability (std is meaningful here)
    summarize(results, group_by='algorithm', metrics=QUALITY_METRICS)

CLI (auto-detects swept parameters, produces all relevant tables):
    python analysis/tables.py --study_dir results/my_study
    python analysis/tables.py --study_dir results/my_study --group_by lmbda
"""

import sys
import os
import csv
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .load import load_results, filter_results
except ImportError:
    from analysis.load import load_results, filter_results


# ── metric definitions ────────────────────────────────────────────────────────
# Extend these dicts to add new metrics — nothing else needs to change.

QUALITY_METRICS = {
    'PSNR_mean':  'PSNR',
    'SSIM_mean':  'SSIM',
    'SAM_mean':   'SAM',
    'RNMSE_mean': 'RNMSE',
}

CONVERGENCE_METRICS = {
    'convergence_converged_fraction': 'Conv. fraction',
    'convergence_iteration_median':   'Iter. (median)',
    'convergence_final_loss_mean':    'Final loss',
    'total_time':                     'Time (s)',
}

# Parameters considered as candidate grouping axes.
# Add new study parameters here to extend auto-detection.
CANDIDATE_GROUPS = ['algorithm', 'lmbda', 'max_iter_cp', 'noise_level', 'scale', 'p', 'q', 'r']


# ── core primitive ────────────────────────────────────────────────────────────

def summarize(results, group_by, metrics, output_dir=None, name=None):
    """
    Print and optionally save a summary table: mean (± std) of each metric,
    one row per unique value of group_by.

    std is shown only when > 0, i.e. when multiple experiments share the same
    group key (Monte Carlo). For single-experiment groups it shows the mean only.

    Args:
        results:    list of result dicts from load_results()
        group_by:   parameter name to group by (e.g. 'algorithm', 'lmbda')
        metrics:    dict of {zarr_attr_key: display_name} — use QUALITY_METRICS
                    or CONVERGENCE_METRICS, or build a custom dict
        output_dir: if given, saves a CSV to output_dir/{name}_by_{group_by}.csv
        name:       table label, used in title and filename ('quality' / 'convergence')
    """
    groups = {}
    for r in results:
        groups.setdefault(r.get(group_by, 'unknown'), []).append(r)

    rows = []
    for key in sorted(groups.keys(), key=lambda x: (isinstance(x, str), x)):
        row = {group_by: key}
        for attr, label in metrics.items():
            vals = [r[attr] for r in groups[key] if attr in r]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std  = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            row[label] = f'{mean:.4f} ± {std:.4f}' if std > 1e-8 else f'{mean:.4f}'
        rows.append(row)

    title = f'{name or "metrics"} by {group_by}'
    _print_table(rows, title)

    if output_dir and rows:
        tag = name or 'table'
        _save_csv(rows, Path(output_dir) / f'{tag}_by_{group_by}.csv')

    return rows


# ── formatting ────────────────────────────────────────────────────────────────

def _print_table(rows, title):
    if not rows:
        print(f'\n{title}: (no data)')
        return
    cols   = list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(str(r.get(c, ''))) for r in rows)) for c in cols}
    hr     = '  '.join('-' * widths[c] for c in cols)
    header = '  '.join(str(c).ljust(widths[c]) for c in cols)
    print(f'\n{title}')
    print(header)
    print(hr)
    for row in rows:
        print('  '.join(str(row.get(c, '')).ljust(widths[c]) for c in cols))


def _save_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved: {path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Metric summary tables for a study')
    parser.add_argument('--study_dir',   required=True, help='Directory produced by a study_*.sh script')
    parser.add_argument('--output_dir',  default=None,  help='Output directory (default: <study_dir>/tables)')
    parser.add_argument('--group_by',    default=None,  help='Override auto-detection (e.g. lmbda, algorithm)')
    parser.add_argument('--lmbda',       type=float,    default=None, help='Fix λ before summarizing')
    parser.add_argument('--max_iter_cp', type=int,      default=None, help='Fix CP iterations before summarizing')
    parser.add_argument('--algorithm',   default=None,               help='Fix algorithm before summarizing')
    args = parser.parse_args()

    output_dir = args.output_dir or f'{args.study_dir}/tables'
    results = load_results(args.study_dir)

    filters = {k: v for k, v in [('lmbda', args.lmbda), ('max_iter_cp', args.max_iter_cp),
                                   ('algorithm', args.algorithm)] if v is not None}
    if filters:
        results = filter_results(results, **filters)

    groups = [args.group_by] if args.group_by else [
        p for p in CANDIDATE_GROUPS
        if len({r.get(p) for r in results if p in r}) > 1
    ]

    if not groups:
        print('No parameter varies across experiments — nothing to group by.')
        return

    for group_by in groups:
        summarize(results, group_by, QUALITY_METRICS,     output_dir, name='quality')
        summarize(results, group_by, CONVERGENCE_METRICS, output_dir, name='convergence')

    print(f'\nTables saved to: {output_dir}')


if __name__ == '__main__':
    main()
