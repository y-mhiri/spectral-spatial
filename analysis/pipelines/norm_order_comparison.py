#!/usr/bin/env python3
"""
Pipeline 4: Norm Order Comparison
Answers: "How do different norm orders (p, q, r) affect algorithm performance?"
"""

import os
import json
import numpy as np
from typing import Dict, List, Any

from analysis.data_io import load_experiment_data
from analysis.plots import plot_parameter_impact
from analysis.pipeline_utils import load_study_results, extract_metric


def norm_order_comparison_pipeline(study_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Compare performance across different norm order combinations.
    
    Args:
        study_dir: Directory containing norm order sweep results
        output_dir: Directory to save analysis results
    
    Returns:
        Dictionary with comparison summary
    """
    print("=== Norm Order Comparison Pipeline ===")
    print(f"Input: {study_dir}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all experiment results
    print("\nLoading experiment results...")
    results = load_study_results(study_dir)
    
    # Extract norm orders and metrics
    print("\nExtracting data...")
    
    # Group by norm orders (p, q, r)
    norm_groups = {}
    for result in results:
        params = result.get('parameters', {})
        p = params.get('p', 'unknown')
        q = params.get('q', 'unknown')
        r = params.get('r', 'unknown')
        
        norm_key = f"p{p}_q{q}_r{r}"
        if norm_key not in norm_groups:
            norm_groups[norm_key] = {
                'params': params,
                'metrics': [],
                'convergence': []
            }
        
        # Extract metrics
        psnr = extract_metric(result, 'PSNR')
        ssim = extract_metric(result, 'SSIM')
        rmse = extract_metric(result, 'RNMSE')
        sam = extract_metric(result, 'SAM')
        
        norm_groups[norm_key]['metrics'].append({
            'PSNR': psnr,
            'SSIM': ssim,
            'RNMSE': rmse,
            'SAM': sam
        })
        
        norm_groups[norm_key]['convergence'].append({
            'converged': result.get('convergence_converged', False),
            'iteration': result.get('convergence_iteration', 0),
            'final_loss': result.get('convergence_final_loss', 0)
        })
    
    # Create sorted list of norm keys
    norm_keys = sorted(norm_groups.keys())
    
    # Generate figures
    print("\nGenerating figures...")
    
    # Figure 1: PSNR vs Norm Order
    psnr_values = [np.mean([m['PSNR'] for m in norm_groups[key]['metrics']]) for key in norm_keys]
    plot_parameter_impact(
        param_values=norm_keys,
        metric_values=psnr_values,
        param_name="Norm Order",
        metric_name="PSNR (dB)",
        output_dir=output_dir,
        title="PSNR vs Norm Order"
    )
    
    # Figure 2: SSIM vs Norm Order
    ssim_values = [np.mean([m['SSIM'] for m in norm_groups[key]['metrics']]) for key in norm_keys]
    plot_parameter_impact(
        param_values=norm_keys,
        metric_values=ssim_values,
        param_name="Norm Order",
        metric_name="SSIM",
        output_dir=output_dir,
        title="SSIM vs Norm Order"
    )
    
    # Figure 3: RMSE vs Norm Order
    rmse_values = [np.mean([m['RNMSE'] for m in norm_groups[key]['metrics']]) for key in norm_keys]
    plot_parameter_impact(
        param_values=norm_keys,
        metric_values=rmse_values,
        param_name="Norm Order",
        metric_name="RNMSE",
        output_dir=output_dir,
        title="RMSE vs Norm Order"
    )
    
    # Figure 4: SAM vs Norm Order
    sam_values = [np.mean([m['SAM'] for m in norm_groups[key]['metrics']]) for key in norm_keys]
    plot_parameter_impact(
        param_values=norm_keys,
        metric_values=sam_values,
        param_name="Norm Order",
        metric_name="SAM (degrees)",
        output_dir=output_dir,
        title="SAM vs Norm Order"
    )
    
    # Figure 5: Convergence rate vs Norm Order
    plt.figure(figsize=(12, 6))
    convergence_rates = [np.mean([c['converged'] for c in norm_groups[key]['convergence']]) for key in norm_keys]
    plt.bar(range(len(norm_keys)), convergence_rates, color='skyblue', alpha=0.7)
    plt.xticks(range(len(norm_keys)), norm_keys, rotation=45, ha='right')
    plt.ylabel('Convergence Rate', fontsize=14)
    plt.title('Convergence Success vs Norm Order', fontsize=16, pad=20)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence_rate.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/convergence_rate.png")
    
    # Figure 6: Final loss vs Norm Order
    plt.figure(figsize=(12, 6))
    final_losses = [np.mean([c['final_loss'] for c in norm_groups[key]['convergence']]) for key in norm_keys]
    plt.bar(range(len(norm_keys)), final_losses, color='salmon', alpha=0.7)
    plt.xticks(range(len(norm_keys)), norm_keys, rotation=45, ha='right')
    plt.ylabel('Average Final Loss', fontsize=14)
    plt.title('Final Loss vs Norm Order', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/final_loss.png")
    
    # Generate summary
    print("\nGenerating summary...")
    summary = {
        'norm_orders': norm_keys,
        'metrics': {},
        'convergence': {}
    }
    
    for i, key in enumerate(norm_keys):
        group = norm_groups[key]
        summary['metrics'][key] = {
            'PSNR': {
                'mean': float(np.mean([m['PSNR'] for m in group['metrics']])),
                'std': float(np.std([m['PSNR'] for m in group['metrics']])),
                'min': float(np.min([m['PSNR'] for m in group['metrics']])),
                'max': float(np.max([m['PSNR'] for m in group['metrics']]))
            },
            'SSIM': {
                'mean': float(np.mean([m['SSIM'] for m in group['metrics']])),
                'std': float(np.std([m['SSIM'] for m in group['metrics']])),
                'min': float(np.min([m['SSIM'] for m in group['metrics']])),
                'max': float(np.max([m['SSIM'] for m in group['metrics']]))
            },
            'RNMSE': {
                'mean': float(np.mean([m['RNMSE'] for m in group['metrics']])),
                'std': float(np.std([m['RNMSE'] for m in group['metrics']])),
                'min': float(np.min([m['RNMSE'] for m in group['metrics']])),
                'max': float(np.max([m['RNMSE'] for m in group['metrics']]))
            },
            'SAM': {
                'mean': float(np.mean([m['SAM'] for m in group['metrics']])),
                'std': float(np.std([m['SAM'] for m in group['metrics']])),
                'min': float(np.min([m['SAM'] for m in group['metrics']])),
                'max': float(np.max([m['SAM'] for m in group['metrics']]))
            }
        }
        summary['convergence'][key] = {
            'rate': float(np.mean([c['converged'] for c in group['convergence']])),
            'avg_iteration': float(np.mean([c['iteration'] for c in group['convergence']])),
            'avg_final_loss': float(np.mean([c['final_loss'] for c in group['convergence']]))
        }
    
    # Find best norm order for each metric
    best_norm = {
        'PSNR': max(norm_keys, key=lambda k: summary['metrics'][k]['PSNR']['mean']),
        'SSIM': max(norm_keys, key=lambda k: summary['metrics'][k]['SSIM']['mean']),
        'RNMSE': min(norm_keys, key=lambda k: summary['metrics'][k]['RNMSE']['mean']),
        'SAM': min(norm_keys, key=lambda k: summary['metrics'][k]['SAM']['mean']),
        'convergence': max(norm_keys, key=lambda k: summary['convergence'][k]['rate'])
    }
    summary['best_norm'] = best_norm
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Pipeline completed!")
    print(f"Summary saved to: {output_dir}/summary.json")
    print(f"Figures saved to: {output_dir}/")
    print(f"\nBest norm orders found:")
    for metric, norm_key in best_norm.items():
        print(f"  {metric}: {norm_key}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Norm Order Comparison Pipeline"
    )
    parser.add_argument("--study_dir", type=str, required=True,
                       help="Directory containing norm order sweep results")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    norm_order_comparison_pipeline(args.study_dir, args.output_dir)
