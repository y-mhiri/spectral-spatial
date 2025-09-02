#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.append(os.path.dirname(__file__))
from loaders import *

def quick_summary(experiment_dir):
    """Generate quick experiment summary"""
    print("="*60)
    print(" EXPERIMENT SUMMARY")
    print("="*60)
    
    # Validate structure
    valid, msg = validate_experiment_structure(experiment_dir)
    if not valid:
        print(f"ERROR: {msg}")
        return False
    
    # Load metadata
    all_metadata = load_experiment_metadata(experiment_dir)
    completion = check_experiment_completion(experiment_dir)
    
    print(f"Total Groups: {completion['total_groups']}")
    print(f"Successful: {completion['successful_groups']} ({completion['success_rate']*100:.1f}%)")
    print(f"Failed: {len(completion['failed_groups'])}")
    
    if not all_metadata:
        print("No successful runs found")
        return False
    
    # Parameter ranges
    print(f"\nPARAMETER RANGES:")
    for param in ['algorithm', 'lambda', 'lambda_m', 'noise_level', 'max_iter']:
        values = get_unique_parameter_values(all_metadata, param)
        if values:
            if len(values) > 10:
                print(f"  {param}: {len(values)} values [{min(values)} to {max(values)}]")
            else:
                print(f"  {param}: {values}")
    
    # Metrics summary
    print(f"\nMETRICS SUMMARY:")
    for metric in ['psnr', 'ssim', 'sam']:
        stats = extract_metric_statistics(all_metadata, metric)
        if stats:
            print(f"  {metric.upper()}: {stats['mean']:.3f}±{stats['std']:.3f} [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    # Timing
    times = [extract_parameter_value(m, 'total_time', 0) for m in all_metadata]
    if any(t > 0 for t in times):
        print(f"  TIME: {np.mean(times):.2f}±{np.std(times):.2f}s per group")
    
    # Memory estimate
    mem_est = get_memory_usage_estimate(all_metadata)
    print(f"\nMEMORY ESTIMATE: {mem_est['estimated_gb']:.2f} GB if all arrays loaded")
    print(f"\nMEMORY ESTIMATE: {mem_est['estimated_mb']:.2f} MB if all arrays loaded")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Quick experiment summary')
    parser.add_argument('--storage_path', type=str, required=True)
    parser.add_argument('--experiment_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.experiment_dir is None:
        experiment_dir = os.path.join(*args.storage_path.split('/')[0:-1])
    else:
        experiment_dir = args.experiment_dir

    success = quick_summary(experiment_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()