#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.append(os.path.dirname(__file__))
from loaders import *
from plot_helpers import *
from scipy import stats as sp_stats

def compare_groups(experiment_dir, param_name, param_values, metric='psnr'):
    """Compare specific parameter groups"""
    print("="*60)
    print(f" GROUP COMPARISON: {param_name}")
    print("="*60)
    
    # Load data
    all_metadata = load_experiment_metadata(experiment_dir)
    successful = filter_successful_runs(all_metadata)
    
    # Filter groups
    groups_data = {}
    for value in param_values:
        filtered = filter_by_parameter(successful, param_name, value)
        if filtered:
            groups_data[str(value)] = filtered
    
    if len(groups_data) < 2:
        print("Need at least 2 groups to compare")
        return False
    
    # Compare groups
    print(f"COMPARISON RESULTS ({metric.upper()}):")
    print("-"*50)
    
    comparison_data = []
    for group_name, group_metadata in groups_data.items():
        stats = extract_metric_statistics(group_metadata, metric)
        if stats:
            comparison_data.append({
                'group': group_name,
                'mean': stats['mean'],
                'std': stats['std'],
                'count': stats['count'],
                'values': stats['values']
            })
            print(f"{group_name:10s}: {stats['mean']:.4f}±{stats['std']:.4f} (n={stats['count']})")
    
    # Statistical tests
    if len(comparison_data) == 2:
        group1, group2 = comparison_data[0], comparison_data[1]
        
        # t-test
        t_stat, p_value = sp_stats.ttest_ind(group1['values'], group2['values'])
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((group1['std']**2 + group2['std']**2) / 2)
        cohens_d = (group1['mean'] - group2['mean']) / pooled_std
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"\nSTATISTICAL TEST:")
        print(f"  t-test: t={t_stat:.3f}, p={p_value:.4f} {significance}")
        print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
    
    # Create box plot
    output_dir = os.path.join(experiment_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    plot_data = []
    for item in comparison_data:
        for value in item['values']:
            plot_data.append({'group': item['group'], metric: value})
    
    plot_df = pd.DataFrame(plot_data)
    plot_path = os.path.join(output_dir, f'comparison_{param_name}_{metric}.png')
    
    create_box_plot(plot_df, 'group', metric, 
                   title=f'{metric.upper()} Comparison by {param_name}',
                   output_path=plot_path)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Compare parameter groups')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--param_name', type=str, required=True)
    parser.add_argument('--param_values', nargs='+', required=True, 
                       help='Parameter values to compare')
    parser.add_argument('--metric', type=str, default='psnr')
    
    args = parser.parse_args()
    
    # Convert param_values to appropriate type
    param_values = []
    for val in args.param_values:
        try:
            if '.' in val or 'e' in val.lower():
                param_values.append(float(val))
            else:
                param_values.append(int(val))
        except ValueError:
            param_values.append(val)  # Keep as string
    
    success = compare_groups(args.experiment_dir, args.param_name, param_values, args.metric)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()