#!/usr/bin/env python3
"""
Pipeline 1: Convergence Analysis
Answers: "Depending on Chambolle-Pock iterations and regularization, does the algorithm converge?"
"""

import os
import json
import numpy as np
from glob import glob
from typing import Dict, List, Any

from analysis.data_io import load_experiment_data
from analysis.plots import plot_convergence_comparison, plot_parameter_impact


def convergence_analysis_pipeline(study_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Analyze convergence behavior across parameter sweeps.
    
    Args:
        study_dir: Directory containing sweep results
        output_dir: Directory to save analysis results
    
    Returns:
        Dictionary with analysis summary
    """
    print("=== Convergence Analysis Pipeline ===")
    print(f"Input: {study_dir}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all experiment results
    print("\nLoading experiment results...")
    zarr_files = sorted(glob(f"{study_dir}/run_*/results.zarr"))
    
    if not zarr_files:
        raise ValueError(f"No results found in {study_dir}")
    
    results = []
    for zarr_file in zarr_files:
        try:
            data = load_experiment_data(zarr_file)
            results.append(data)
            print(f"  ✓ Loaded: {os.path.basename(zarr_file)}")
        except Exception as e:
            print(f"  ✗ Failed: {zarr_file} - {e}")
    
    if not results:
        raise ValueError("No valid results loaded")
    
    # Extract parameters and metrics
    print("\nExtracting data...")
    
    # Group by parameters of interest
    param_groups = {}
    for result in results:
        params = result.get('parameters', {})
        
        # Key parameters for convergence analysis
        lmbda = params.get('lmbda', 'unknown')
        max_iter_cp = params.get('max_iter_cp', 'unknown')
        
        key = f"lmbda{lmbda}_cp{max_iter_cp}"
        if key not in param_groups:
            param_groups[key] = {
                'params': params,
                'loss_curves': [],
                'convergence': [],
                'metrics': []
            }
        
        param_groups[key]['loss_curves'].append(result['loss'][0])  # First image
        param_groups[key]['convergence'].append({
            'converged': result.get('convergence_converged', False),
            'iteration': result.get('convergence_iteration', 0),
            'final_loss': result.get('convergence_final_loss', 0)
        })
        param_groups[key]['metrics'].append({
            'PSNR': result.get('metrics', {}).get('PSNR', [0])[0] if isinstance(result.get('metrics', {}).get('PSNR', [0]), list) else result.get('metrics', {}).get('PSNR', 0),
            'SSIM': result.get('metrics', {}).get('SSIM', [0])[0] if isinstance(result.get('metrics', {}).get('SSIM', [0]), list) else result.get('metrics', {}).get('SSIM', 0)
        })
    
    # Generate figures
    print("\nGenerating figures...")
    
    # Figure 1: Convergence curves by lambda
    lmbda_groups = {}
    for key, data in param_groups.items():
        lmbda = data['params'].get('lmbda', 'unknown')
        if lmbda not in lmbda_groups:
            lmbda_groups[lmbda] = []
        lmbda_groups[lmbda].append(np.mean(data['loss_curves'], axis=0))
    
    if len(lmbda_groups) > 1:
        plot_convergence_comparison(
            loss_curves=list(lmbda_groups.values()),
            param_values=list(lmbda_groups.keys()),
            param_name="λ (lmbda)",
            output_dir=output_dir,
            title="Convergence: Regularization Impact"
        )
    
    # Figure 2: Convergence curves by CP iterations
    cp_groups = {}
    for key, data in param_groups.items():
        cp_iter = data['params'].get('max_iter_cp', 'unknown')
        if cp_iter not in cp_groups:
            cp_groups[cp_iter] = []
        cp_groups[cp_iter].append(np.mean(data['loss_curves'], axis=0))
    
    if len(cp_groups) > 1:
        plot_convergence_comparison(
            loss_curves=list(cp_groups.values()),
            param_values=list(cp_groups.keys()),
            param_name="CP iterations",
            output_dir=output_dir,
            title="Convergence: CP Iterations Impact"
        )
    
    # Figure 3: Convergence iteration vs parameters
    plt.figure(figsize=(12, 6))
    
    for key, data in param_groups.items():
        lmbda = data['params'].get('lmbda', 'unknown')
        cp_iter = data['params'].get('max_iter_cp', 'unknown')
        avg_conv_iter = np.mean([c['iteration'] for c in data['convergence']])
        label = f"λ={lmbda}, CP={cp_iter}"
        
        # Use color to indicate convergence success
        color = 'green' if np.mean([c['converged'] for c in data['convergence']]) > 0.5 else 'red'
        plt.scatter(cp_iter, avg_conv_iter, color=color, s=100, label=label, alpha=0.7)
    
    plt.xlabel('Chambolle-Pock Iterations', fontsize=14)
    plt.ylabel('Average Convergence Iteration', fontsize=14)
    plt.title('Convergence Speed: Parameter Impact', fontsize=16, pad=20)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence_speed.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/convergence_speed.png")
    
    # Figure 4: Final loss vs parameters
    plt.figure(figsize=(12, 6))
    
    for key, data in param_groups.items():
        lmbda = data['params'].get('lmbda', 'unknown')
        cp_iter = data['params'].get('max_iter_cp', 'unknown')
        avg_final_loss = np.mean([c['final_loss'] for c in data['convergence']])
        label = f"λ={lmbda}, CP={cp_iter}"
        
        color = 'green' if avg_final_loss < 0.1 else 'orange' if avg_final_loss < 0.5 else 'red'
        plt.scatter(cp_iter, avg_final_loss, color=color, s=100, label=label, alpha=0.7)
    
    plt.xlabel('Chambolle-Pock Iterations', fontsize=14)
    plt.ylabel('Average Final Loss', fontsize=14)
    plt.title('Final Loss: Parameter Impact', fontsize=16, pad=20)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/final_loss.png")
    
    # Generate summary
    print("\nGenerating summary...")
    summary = {
        'parameter_groups': {},
        'overall': {
            'total_runs': len(results),
            'convergence_rate': np.mean([r.get('convergence_converged', False) for r in results]),
            'avg_final_loss': np.mean([r.get('convergence_final_loss', 0) for r in results]),
            'avg_psnr': np.mean([(r.get('metrics', {}).get('PSNR', [0])[0] if isinstance(r.get('metrics', {}).get('PSNR', [0]), list) else r.get('metrics', {}).get('PSNR', 0)) for r in results]),
            'avg_ssim': np.mean([(r.get('metrics', {}).get('SSIM', [0])[0] if isinstance(r.get('metrics', {}).get('SSIM', [0]), list) else r.get('metrics', {}).get('SSIM', 0)) for r in results])
        }
    }
    
    for key, data in param_groups.items():
        summary['parameter_groups'][key] = {
            'parameters': data['params'],
            'convergence_rate': np.mean([c['converged'] for c in data['convergence']]),
            'avg_convergence_iter': np.mean([c['iteration'] for c in data['convergence']]),
            'avg_final_loss': np.mean([c['final_loss'] for c in data['convergence']]),
            'avg_psnr': np.mean([m['PSNR'] for m in data['metrics']]),
            'avg_ssim': np.mean([m['SSIM'] for m in data['metrics']])
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
        description="Convergence Analysis Pipeline"
    )
    parser.add_argument("--study_dir", type=str, required=True,
                       help="Directory containing sweep results")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    convergence_analysis_pipeline(args.study_dir, args.output_dir)
