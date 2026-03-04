#!/usr/bin/env python3
"""
Pipeline 3: Algorithm Comparison
Answers: "How do CTV and GradAlign compare under different conditions?"
"""

import os
import json
import numpy as np
from glob import glob
from typing import Dict, List, Any

from analysis.data_io import load_experiment_data, compare_experiment_results
from analysis.plots import plot_algorithm_comparison, plot_parameter_impact


def algorithm_comparison_pipeline(study_dirs: Dict[str, str], output_dir: str) -> Dict[str, Any]:
    """
    Compare algorithms across different experimental conditions.
    
    Args:
        study_dirs: {algorithm_name: study_directory}
        output_dir: Directory to save analysis results
    
    Returns:
        Dictionary with comparison summary
    """
    print("=== Algorithm Comparison Pipeline ===")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results for each algorithm
    print("\nLoading experiment results...")
    algorithm_results = {}
    
    for algorithm, study_dir in study_dirs.items():
        print(f"\n{algorithm}:")
        zarr_files = sorted(glob(f"{study_dir}/run_*/results.zarr"))
        
        if not zarr_files:
            print(f"  ✗ No results found in {study_dir}")
            continue
        
        results = []
        for zarr_file in zarr_files:
            try:
                data = load_experiment_data(zarr_file)
                results.append(data)
                print(f"  ✓ Loaded: {os.path.basename(zarr_file)}")
            except Exception as e:
                print(f"  ✗ Failed: {zarr_file} - {e}")
        
        if results:
            algorithm_results[algorithm] = results
    
    if not algorithm_results:
        raise ValueError("No valid results loaded for any algorithm")
    
    # Extract metrics for comparison
    print("\nExtracting metrics for comparison...")
    
    # Common metrics to compare
    metrics = ['PSNR', 'SSIM', 'RNMSE', 'SAM']
    comparison_summary = {}
    
    for metric in metrics:
        print(f"\nComparing {metric}...")
        
        # Collect all values for each algorithm
        algorithm_metric_values = {}
        for algorithm, results in algorithm_results.items():
            values = []
            for result in results:
                metric_val = result.get('metrics', {}).get(metric, [0])
                if isinstance(metric_val, list) and len(metric_val) > 0:
                    values.append(metric_val[0])
                elif not isinstance(metric_val, list):
                    values.append(metric_val)
            
            if values:
                algorithm_metric_values[algorithm] = values
                print(f"  {algorithm}: {len(values)} samples, mean={np.mean(values):.3f} ± {np.std(values):.3f}")
        
        if len(algorithm_metric_values) >= 2:
            # Generate comparison plot
            plot_algorithm_comparison(
                algorithm_results=algorithm_metric_values,
                metric_name=metric,
                output_dir=output_dir,
                title=f"{metric} Comparison"
            )
            
            # Store comparison statistics
            comparison_summary[metric] = compare_experiment_results(
                results_list=[{'metrics': {metric: vals}} for vals in algorithm_metric_values.values()],
                metric=metric
            )
    
    # Generate additional comparison figures
    print("\nGenerating additional figures...")
    
    # Figure: Convergence rate comparison
    plt.figure(figsize=(10, 6))
    convergence_rates = {}
    for algorithm, results in algorithm_results.items():
        converged = [r.get('convergence_converged', False) for r in results]
        convergence_rates[algorithm] = np.mean(converged) if converged else 0
    
    algorithms = list(convergence_rates.keys())
    rates = list(convergence_rates.values())
    
    bars = plt.bar(algorithms, rates, color=['skyblue', 'salmon'])
    plt.bar_label(bars, fmt='%.2f', padding=3)
    plt.ylabel('Convergence Rate', fontsize=14)
    plt.title('Convergence Rate Comparison', fontsize=16, pad=20)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/convergence_rate_comparison.png")
    
    # Figure: Final loss comparison
    plt.figure(figsize=(10, 6))
    final_losses = {}
    for algorithm, results in algorithm_results.items():
        losses = [r.get('convergence_final_loss', 0) for r in results]
        final_losses[algorithm] = np.mean(losses) if losses else 0
    
    algorithms = list(final_losses.keys())
    losses = list(final_losses.values())
    
    bars = plt.bar(algorithms, losses, color=['skyblue', 'salmon'])
    plt.bar_label(bars, fmt='%.3f', padding=3)
    plt.ylabel('Average Final Loss', fontsize=14)
    plt.title('Final Loss Comparison', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_loss_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/final_loss_comparison.png")
    
    # Figure: Computation time comparison (if available)
    try:
        plt.figure(figsize=(10, 6))
        computation_times = {}
        for algorithm, results in algorithm_results.items():
            # Note: This would require storing computation time in results
            # For now, we'll skip this as it's not currently stored
            pass
        plt.close()
    except Exception as e:
        print(f"Skipping computation time comparison: {e}")
    
    # Generate summary
    print("\nGenerating summary...")
    summary = {
        'algorithms': list(algorithm_results.keys()),
        'metric_comparisons': comparison_summary,
        'convergence_rates': convergence_rates,
        'final_losses': final_losses,
        'sample_counts': {alg: len(results) for alg, results in algorithm_results.items()}
    }
    
    # Add relative performance
    if len(algorithm_results) >= 2:
        ref_algorithm = list(algorithm_results.keys())[0]
        ref_psnr = comparison_summary['PSNR']['values'][0]
        
        for i, algorithm in enumerate(list(algorithm_results.keys())[1:]):
            current_psnr = comparison_summary['PSNR']['values'][i+1]
            summary[f'relative_improvement_{algorithm}'] = {
                'PSNR': current_psnr - ref_psnr,
                'percentage': ((current_psnr - ref_psnr) / ref_psnr * 100) if ref_psnr != 0 else 0
            }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Pipeline completed!")
    print(f"Summary saved to: {output_dir}/summary.json")
    print(f"Figures saved to: {output_dir}/")
    
    return summary


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Algorithm Comparison Pipeline"
    )
    parser.add_argument("--config", type=str, required=True,
                       help="JSON file with {algorithm: study_dir} mapping")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        study_dirs = json.load(f)
    
    algorithm_comparison_pipeline(study_dirs, args.output_dir)
