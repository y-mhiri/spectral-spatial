#!/usr/bin/env python3
"""
Master pipeline runner for scientific analysis.
Runs all three pipelines to answer the research questions.
"""

import subprocess
import json
import os
from datetime import datetime


def run_convergence_analysis(study_dir: str, output_dir: str) -> None:
    """Run convergence analysis pipeline"""
    print("=" * 60)
    print("RUNNING: Convergence Analysis Pipeline")
    print("=" * 60)
    
    cmd = [
        "python", "analysis/pipelines/convergence_analysis.py",
        "--study_dir", study_dir,
        "--output_dir", output_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Convergence analysis completed successfully")
        print(f"   Results: {output_dir}/")
    else:
        print("❌ Convergence analysis failed")
        print(f"   Error: {result.stderr}")
    
    print()


def run_noise_impact_analysis(study_dir: str, output_dir: str) -> None:
    """Run noise impact analysis pipeline"""
    print("=" * 60)
    print("RUNNING: Noise Impact Analysis Pipeline")
    print("=" * 60)
    
    cmd = [
        "python", "analysis/pipelines/noise_impact.py",
        "--study_dir", study_dir,
        "--output_dir", output_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Noise impact analysis completed successfully")
        print(f"   Results: {output_dir}/")
    else:
        print("❌ Noise impact analysis failed")
        print(f"   Error: {result.stderr}")
    
    print()


def run_algorithm_comparison(ctv_dir: str, gradalign_dir: str, output_dir: str) -> None:
    """Run algorithm comparison pipeline"""
    print("=" * 60)
    print("RUNNING: Algorithm Comparison Pipeline")
    print("=" * 60)
    
    # Create config file
    config = {
        "CTV": ctv_dir,
        "GradAlign": gradalign_dir
    }
    
    config_file = f"{output_dir}/comparison_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    cmd = [
        "python", "analysis/pipelines/algorithm_comparison.py",
        "--config", config_file,
        "--output_dir", output_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Algorithm comparison completed successfully")
        print(f"   Results: {output_dir}/")
    else:
        print("❌ Algorithm comparison failed")
        print(f"   Error: {result.stderr}")
    
    print()


def run_norm_order_comparison(study_dir: str, output_dir: str) -> None:
    """Run norm order comparison pipeline"""
    print("=" * 60)
    print("RUNNING: Norm Order Comparison Pipeline")
    print("=" * 60)
    
    cmd = [
        "python", "analysis/pipelines/norm_order_comparison.py",
        "--study_dir", study_dir,
        "--output_dir", output_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Norm order comparison completed successfully")
        print(f"   Results: {output_dir}/")
    else:
        print("❌ Norm order comparison failed")
        print(f"   Error: {result.stderr}")
    
    print()


def run_all_pipelines(
    convergence_study_dir: str,
    noise_study_dir: str,
    norm_study_dir: str,
    ctv_study_dir: str,
    gradalign_study_dir: str,
    base_output_dir: str
) -> None:
    """
    Run all scientific pipelines.
    
    Args:
        convergence_study_dir: Directory with convergence study results
        noise_study_dir: Directory with noise study results
        norm_study_dir: Directory with norm order study results
        ctv_study_dir: Directory with CTV algorithm results
        gradalign_study_dir: Directory with GradAlign algorithm results
        base_output_dir: Base directory for all pipeline outputs
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_output_dir = os.path.join(base_output_dir, f"analysis_{timestamp}")
    os.makedirs(master_output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SCIENTIFIC ANALYSIS PIPELINE")
    print(f"Started: {timestamp}")
    print("=" * 60)
    print()
    
    # Run each pipeline
    run_convergence_analysis(
        convergence_study_dir,
        os.path.join(master_output_dir, "convergence_analysis")
    )
    
    run_noise_impact_analysis(
        noise_study_dir,
        os.path.join(master_output_dir, "noise_impact")
    )
    
    # Norm order comparison removed for simplicity
    # if os.path.exists(norm_study_dir):
    #     run_norm_order_comparison(
    #         norm_study_dir,
    #         os.path.join(master_output_dir, "norm_order_comparison")
    #     )
    else:
        print("Norm order study directory not found. Skipping norm order comparison.")
    
    run_algorithm_comparison(
        ctv_study_dir,
        gradalign_study_dir,
        os.path.join(master_output_dir, "algorithm_comparison")
    )
    
    # Generate master summary
    print("=" * 60)
    print("GENERATING MASTER SUMMARY")
    print("=" * 60)
    
    # Update master summary
    pipelines = [
        {
            'name': 'Convergence Analysis',
            'input': convergence_study_dir,
            'output': os.path.join(master_output_dir, "convergence_analysis")
        },
        {
            'name': 'Noise Impact Analysis',
            'input': noise_study_dir,
            'output': os.path.join(master_output_dir, "noise_impact")
        },
        {
            'name': 'Algorithm Comparison',
            'input': {'CTV': ctv_study_dir, 'GradAlign': gradalign_study_dir},
            'output': os.path.join(master_output_dir, "algorithm_comparison")
        }
    ]
    
    master_summary = {
        'timestamp': timestamp,
        'pipelines': pipelines,
        'output_directory': master_output_dir
    }
    
    summary_file = os.path.join(master_output_dir, "master_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(master_summary, f, indent=2)
    
    print(f"✅ Master summary saved to: {summary_file}")
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print("Results available in:")
    print(f"  {master_output_dir}/")
    print()
    print("Individual pipeline outputs:")
    print(f"  - Convergence: {master_output_dir}/convergence_analysis/")
    print(f"  - Noise Impact: {master_output_dir}/noise_impact/")
    print(f"  - Algorithm Comparison: {master_output_dir}/algorithm_comparison/")
    print()
    print("Key figures generated:")
    print("  - Convergence curves by parameter")
    print("  - Metric degradation vs noise")
    print("  - Algorithm comparison boxplots")
    print("  - Convergence rate comparisons")
    print("  - Final loss comparisons")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run all scientific analysis pipelines"
    )
    parser.add_argument("--convergence_study", type=str, required=True,
                       help="Directory with convergence study results")
    parser.add_argument("--noise_study", type=str, required=True,
                       help="Directory with noise study results")
    parser.add_argument("--ctv_study", type=str, required=True,
                       help="Directory with CTV algorithm results")
    parser.add_argument("--gradalign_study", type=str, required=True,
                       help="Directory with GradAlign algorithm results")
    parser.add_argument("--output", type=str, default="results/analysis",
                       help="Base output directory")
    
    args = parser.parse_args()
    
    run_all_pipelines(
        args.convergence_study,
        args.noise_study,
        args.norm_study,
        args.ctv_study,
        args.gradalign_study,
        args.output
    )
