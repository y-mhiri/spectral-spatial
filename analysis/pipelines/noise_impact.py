#!/usr/bin/env python3
"""
Pipeline 2: Noise Impact Analysis
Answers: "How does algorithm performance degrade with increasing noise levels?"
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from analysis.data_io import load_experiment_data
from analysis.plots import plot_parameter_impact
from analysis.pipeline_utils import load_study_results, extract_metric


def noise_impact_pipeline(study_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Analyze algorithm performance across noise levels.
    
    Args:
        study_dir: Directory containing sweep results
        output_dir: Directory to save analysis results
    
    Returns:
        Dictionary with analysis summary
    """
    print("=== Noise Impact Analysis Pipeline ===")
    print(f"Input: {study_dir}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all experiment results
    print("\nLoading experiment results...")
    results = load_study_results(study_dir)
    
    # Extract noise levels and metrics
    print("\nExtracting data...")
    
    # Group by noise level
    noise_groups = {}
    for result in results:
        params = result.get('parameters', {})
        noise_level = params.get('noise_level', 'unknown')
        
        if noise_level not in noise_groups:
            noise_groups[noise_level] = {
                'params': params,
                'metrics': [],
                'convergence': []
            }
        
        # Extract metrics
        psnr = extract_metric(result, 'PSNR')
        ssim = extract_metric(result, 'SSIM')
        rmse = extract_metric(result, 'RNMSE')
        sam = extract_metric(result, 'SAM')
        
        noise_groups[noise_level]['metrics'].append({
            'PSNR': psnr,
            'SSIM': ssim,
            'RNMSE': rmse,
            'SAM': sam
        })
        
        noise_groups[noise_level]['convergence'].append({
            'converged': result.get('convergence_converged', False),
            'iteration': result.get('convergence_iteration', 0),
            'final_loss': result.get('convergence_final_loss', 0)
        })
    
    # Sort noise levels numerically
    noise_levels = sorted([float(k) for k in noise_groups.keys()])
    
    # Generate figures
    print("\nGenerating figures...")
    
    # Figure 1: PSNR vs Noise Level
    psnr_values = [np.mean([m['PSNR'] for m in noise_groups[str(nl)]['metrics']]) for nl in noise_levels]
    plot_parameter_impact(
        param_values=noise_levels,
        metric_values=psnr_values,
        param_name="Noise Level",
        metric_name="PSNR (dB)",
        output_dir=output_dir,
        title="Reconstruction Quality vs Noise Level"
    )
    
    # Figure 2: SSIM vs Noise Level
    ssim_values = [np.mean([m['SSIM'] for m in noise_groups[str(nl)]['metrics']]) for nl in noise_levels]
    plot_parameter_impact(
        param_values=noise_levels,
        metric_values=ssim_values,
        param_name="Noise Level",
        metric_name="SSIM",
        output_dir=output_dir,
        title="Structural Similarity vs Noise Level"
    )
    
    # Figure 3: RMSE vs Noise Level
    rmse_values = [np.mean([m['RNMSE'] for m in noise_groups[str(nl)]['metrics']]) for nl in noise_levels]
    plot_parameter_impact(
        param_values=noise_levels,
        metric_values=rmse_values,
        param_name="Noise Level",
        metric_name="RNMSE",
        output_dir=output_dir,
        title="Reconstruction Error vs Noise Level"
    )
    
    # Figure 4: SAM vs Noise Level
    sam_values = [np.mean([m['SAM'] for m in noise_groups[str(nl)]['metrics']]) for nl in noise_levels]
    plot_parameter_impact(
        param_values=noise_levels,
        metric_values=sam_values,
        param_name="Noise Level",
        metric_name="SAM (degrees)",
        output_dir=output_dir,
        title="Spectral Angle Mapper vs Noise Level"
    )
    
    # Figure 5: Convergence success rate vs Noise
    plt.figure(figsize=(10, 6))
    convergence_rates = [np.mean([c['converged'] for c in noise_groups[str(nl)]['convergence']]) for nl in noise_levels]
    plt.plot(noise_levels, convergence_rates, 'o-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Noise Level', fontsize=14)
    plt.ylabel('Convergence Rate', fontsize=14)
    plt.title('Convergence Success vs Noise Level', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence_rate.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/convergence_rate.png")
    
    # Figure 6: Final loss vs Noise
    final_losses = [np.mean([c['final_loss'] for c in noise_groups[str(nl)]['convergence']]) for nl in noise_levels]
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, final_losses, 'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Noise Level', fontsize=14)
    plt.ylabel('Final Loss', fontsize=14)
    plt.title('Final Loss vs Noise Level', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/final_loss.png")
    
    # Generate summary
    print("\nGenerating summary...")
    summary = {
        'noise_levels': noise_levels,
        'metrics': {},
        'convergence': {}
    }
    
    for nl in noise_levels:
        nl_str = str(nl)
        group = noise_groups[nl_str]
        summary['metrics'][nl] = {
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
        summary['convergence'][nl] = {
            'rate': float(np.mean([c['converged'] for c in group['convergence']])),
            'avg_iteration': float(np.mean([c['iteration'] for c in group['convergence']])),
            'avg_final_loss': float(np.mean([c['final_loss'] for c in group['convergence']]))
        }
    
    # Add degradation analysis
    if len(noise_levels) >= 2:
        psnr_degradation = [(summary['metrics'][noise_levels[i]]['PSNR']['mean'] - 
                           summary['metrics'][noise_levels[0]]['PSNR']['mean']) 
                          for i in range(1, len(noise_levels))]
        summary['degradation'] = {
            'PSNR_loss_per_noise_unit': float(np.mean(psnr_degradation) / (noise_levels[-1] - noise_levels[0])) if noise_levels[-1] != noise_levels[0] else 0,
            'relative_PSNR_loss': float((summary['metrics'][noise_levels[-1]]['PSNR']['mean'] - 
                                       summary['metrics'][noise_levels[0]]['PSNR']['mean']) / 
                                      summary['metrics'][noise_levels[0]]['PSNR']['mean']) if noise_levels[0] != 0 else 0
        }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Pipeline completed!")
    print(f"Summary saved to: {output_dir}/summary.json")
    print(f"Figures saved to: {output_dir}/")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Noise Impact Analysis Pipeline"
    )
    parser.add_argument("--study_dir", type=str, required=True,
                       help="Directory containing sweep results")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    noise_impact_pipeline(args.study_dir, args.output_dir)
