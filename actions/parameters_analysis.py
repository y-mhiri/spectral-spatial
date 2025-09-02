#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.append(os.path.dirname(__file__))
from loaders import *
from plot_helpers import *

def analyze_parameters(experiment_dir, param1, param2, metric='psnr', algorithm=None):
    """Analyze parameter effects"""
    print("="*60)
    print(f" PARAMETER ANALYSIS: {param1} vs {param2}")
    print("="*60)
    
    # Load and filter data
    all_metadata = load_experiment_metadata(experiment_dir)
    if algorithm:
        all_metadata = filter_by_algorithm(all_metadata, algorithm)
    
    successful = filter_successful_runs(all_metadata)
    print(f"Analyzing {len(successful)} successful runs")
    
    if len(successful) < 2:
        print("Need at least 2 successful runs")
        return False
    
    # Create parameter grid
    grid_df = create_parameter_grid(successful, param1, param2, metric)
    
    if grid_df.empty:
        print("No valid parameter combinations found")
        return False
    
    # Show best combinations
    best_df = grid_df.sort_values(f'{metric}_mean', ascending=False)
    print(f"\nTOP 5 PARAMETER COMBINATIONS:")
    print("-"*50)
    for i, (_, row) in enumerate(best_df.head().iterrows()):
        print(f"{i+1}. {param1}={row[param1]:.2e}, {param2}={row[param2]:.2f} → "
              f"{metric.upper()}={row[f'{metric}_mean']:.4f}±{row[f'{metric}_std']:.4f}")
    
    # Create heatmap
    output_dir = os.path.join(experiment_dir, 'analysis')
    heatmap_path = os.path.join(output_dir, f'heatmap_{param1}_{param2}_{metric}.png')
    
    create_heatmap(grid_df, param1, param2, f'{metric}_mean', 
                   title=f'{metric.upper()} vs {param1} & {param2}',
                   output_path=heatmap_path)
    
    # Save results
    csv_path = os.path.join(output_dir, f'parameter_grid_{param1}_{param2}.csv')
    os.makedirs(output_dir, exist_ok=True)
    grid_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Parameter importance
    param1_range = grid_df[param1].max() - grid_df[param1].min()
    param2_range = grid_df[param2].max() - grid_df[param2].min()
    metric_range = grid_df[f'{metric}_mean'].max() - grid_df[f'{metric}_mean'].min()
    
    print(f"\nPARAMETER SENSITIVITY:")
    print(f"  {metric.upper()} range: {metric_range:.4f}")
    print(f"  {param1} range: {param1_range:.2e}")
    print(f"  {param2} range: {param2_range:.2f}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Analyze parameter effects')
    parser.add_argument('--experiment_dir', type=str, default=None)
    parser.add_argument('--storage_path', type=str, required=True)
    parser.add_argument('--param1', type=str, required=True, help='First parameter (e.g., lambda)')
    parser.add_argument('--param2', type=str, required=True, help='Second parameter (e.g., lambda_m)')
    parser.add_argument('--metric', type=str, default='PSNR', help='Metric to analyze')
    parser.add_argument('--algorithm', type=str, help='Filter by algorithm')

    if args.experiment_dir is None:
        experiment_dir = os.path.join(*args.storage_path.split('/')[0:-1])
    else:
        experiment_dir = args.experiment_dir

    args = parser.parse_args()
    success = analyze_parameters(experiment_dir, args.param1, args.param2, args.metric, args.algorithm)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()