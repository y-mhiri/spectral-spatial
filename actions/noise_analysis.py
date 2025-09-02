#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.append(os.path.dirname(__file__))
from loaders import *
from plot_helpers import *

def analyze_noise_robustness(experiment_dir, algorithm=None, metrics=None):
    """Analyze performance vs noise levels"""
    if metrics is None:
        metrics = ['psnr', 'ssim']
    
    print("="*60)
    print(" NOISE ROBUSTNESS ANALYSIS")
    print("="*60)
    
    # Load and filter data
    all_metadata = load_experiment_metadata(experiment_dir)
    if algorithm:
        all_metadata = filter_by_algorithm(all_metadata, algorithm)
    
    successful = filter_successful_runs(all_metadata)
    print(f"Analyzing {len(successful)} successful runs")
    
    # Group by noise level
    noise_analysis = aggregate_by_parameters(successful, ['noise_level'], 'psnr')
    
    if noise_analysis.empty:
        print("No noise level data found")
        return False
    
    # Add other metrics
    for metric in metrics:
        if metric != 'psnr':
            metric_data = aggregate_by_parameters(successful, ['noise_level'], metric)
            if not metric_data.empty:
                noise_analysis = noise_analysis.merge(metric_data[['noise_level', f'{metric}_mean', f'{metric}_std']], 
                                                     on='noise_level', how='left')
    
    noise_analysis = noise_analysis.sort_values('noise_level')
    
    # Print noise effects
    print(f"\nNOISE ROBUSTNESS:")
    print("-"*50)
    for _, row in noise_analysis.iterrows():
        print(f"Noise={row['noise_level']:4.0f}: ", end="")
        for metric in metrics:
            if f'{metric}_mean' in row:
                print(f"{metric.upper()}={row[f'{metric}_mean']:.3f}±{row[f'{metric}_std']:.3f} ", end="")
        print()
    
    # Find degradation thresholds
    if 'psnr_mean' in noise_analysis.columns:
        baseline_psnr = noise_analysis['psnr_mean'].max()
        threshold_3db = baseline_psnr - 3
        threshold_5db = baseline_psnr - 5
        
        noise_3db = noise_analysis[noise_analysis['psnr_mean'] <= threshold_3db]['noise_level'].min()
        noise_5db = noise_analysis[noise_analysis['psnr_mean'] <= threshold_5db]['noise_level'].min()
        
        print(f"\nDEGRADATION THRESHOLDS:")
        print(f"  Baseline PSNR: {baseline_psnr:.2f} dB")
        if not pd.isna(noise_3db):
            print(f"  3dB degradation at noise: {noise_3db:.0f}")
        if not pd.isna(noise_5db):
            print(f"  5dB degradation at noise: {noise_5db:.0f}")
    
    # Create plots
    output_dir = os.path.join(experiment_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    for metric in metrics:
        if f'{metric}_mean' in noise_analysis.columns:
            plot_path = os.path.join(output_dir, f'{metric}_vs_noise.png')
            create_degradation_curve(noise_analysis, 'noise_level', f'{metric}_mean',
                                   title=f'{metric.upper()} vs Noise Level',
                                   output_path=plot_path)
    
    # Save results
    csv_path = os.path.join(output_dir, 'noise_robustness.csv')
    noise_analysis.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Analyze noise robustness')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--algorithm', type=str, help='Filter by algorithm')
    parser.add_argument('--metrics', nargs='+', default=['psnr', 'ssim'], 
                       help='Metrics to analyze')
    
    args = parser.parse_args()
    success = analyze_noise_robustness(args.experiment_dir, args.algorithm, args.metrics)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()