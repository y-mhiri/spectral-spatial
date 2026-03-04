#!/usr/bin/env python3
"""
Report generation functions for hyperspectral pansharpening experiments.

Creates comprehensive reports with tables, summaries, and visualizations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def generate_experiment_summary(
    results: Dict,
    output_dir: str,
    algorithm_name: str = "Algorithm"
) -> Path:
    """
    Generate comprehensive experiment summary report.
    
    Args:
        results: Dictionary containing experiment results
        output_dir: Directory to save report
        algorithm_name: Name of algorithm
        
    Returns:
        Path to generated report
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate JSON report
    json_report = generate_json_report(results, algorithm_name)
    json_path = output_dir / "report.json"
    
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    # Generate markdown report
    md_report = generate_markdown_report(results, algorithm_name)
    md_path = output_dir / "report.md"
    
    with open(md_path, 'w') as f:
        f.write(md_report)
    
    # Generate text summary
    txt_report = generate_text_summary(results, algorithm_name)
    txt_path = output_dir / "summary.txt"
    
    with open(txt_path, 'w') as f:
        f.write(txt_report)
    
    return output_dir


def generate_json_report(results: Dict, algorithm_name: str) -> Dict:
    """
    Generate JSON report structure.
    
    Args:
        results: Experiment results
        algorithm_name: Algorithm name
        
    Returns:
        Dictionary with JSON report structure
    """
    report = {
        'experiment': {
            'algorithm': algorithm_name,
            'date': results.get('date', 'N/A'),
            'total_time': results.get('total_time', 'N/A')
        },
        'parameters': results.get('parameters', {}),
        'metrics': {},
        'performance': {}
    }
    
    # Add metrics
    if 'metrics' in results:
        for metric, values in results['metrics'].items():
            if isinstance(values, list):
                report['metrics'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': [float(v) for v in values]
                }
            else:
                report['metrics'][metric] = float(values)
    
    # Add performance data
    if 'performance' in results:
        report['performance'] = {k: float(v) for k, v in results['performance'].items()}
    
    return report


def generate_markdown_report(results: Dict, algorithm_name: str) -> str:
    """
    Generate markdown report.
    
    Args:
        results: Experiment results
        algorithm_name: Algorithm name
        
    Returns:
        Markdown report as string
    """
    report = []
    
    # Header
    report.append(f"# {algorithm_name} Experiment Report")
    report.append(f"\n**Date**: {results.get('date', 'N/A')}")
    report.append(f"\n**Total Time**: {results.get('total_time', 'N/A'):.2f} seconds")
    
    # Parameters
    report.append("\n## Parameters")
    if 'parameters' in results:
        for key, value in results['parameters'].items():
            report.append(f"- **{key}**: {value}")
    
    # Metrics
    report.append("\n## Metrics")
    if 'metrics' in results:
        for metric, values in results['metrics'].items():
            if isinstance(values, list):
                mean_val = np.mean(values)
                std_val = np.std(values)
                report.append(f"- **{metric}**: {mean_val:.4f} ± {std_val:.4f}")
            else:
                report.append(f"- **{metric}**: {values:.4f}")
    
    # Performance
    report.append("\n## Performance")
    if 'performance' in results:
        for key, value in results['performance'].items():
            report.append(f"- **{key}**: {value:.2f}")
    
    return "\n".join(report)


def generate_text_summary(results: Dict, algorithm_name: str) -> str:
    """
    Generate concise text summary.
    
    Args:
        results: Experiment results
        algorithm_name: Algorithm name
        
    Returns:
        Text summary as string
    """
    lines = []
    
    lines.append(f"Algorithm: {algorithm_name}")
    lines.append(f"Date: {results.get('date', 'N/A')}")
    lines.append(f"Total Time: {results.get('total_time', 'N/A'):.2f} seconds")
    
    # Key metrics
    if 'metrics' in results:
        key_metrics = ['PSNR', 'SSIM', 'SAM']
        for metric in key_metrics:
            if metric in results['metrics']:
                values = results['metrics'][metric]
                if isinstance(values, list):
                    mean_val = np.mean(values)
                    lines.append(f"{metric}: {mean_val:.4f}")
                else:
                    lines.append(f"{metric}: {values:.4f}")
    
    return "\n".join(lines)


def create_visualization_index(
    output_dir: str,
    algorithm_name: str,
    image_count: int
) -> Path:
    """
    Create HTML index file for visualization results.
    
    Args:
        output_dir: Directory containing visualizations
        algorithm_name: Algorithm name
        image_count: Number of images processed
        
    Returns:
        Path to generated index file
    """
    output_dir = Path(output_dir)
    index_path = output_dir / "index.html"
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{algorithm_name} Visualization Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .section {{ margin-bottom: 30px; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .image-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .image-card img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>{algorithm_name} Visualization Results</h1>
    
    <div class="section">
        <h2>Convergence Plots</h2>
        <div class="image-grid">
            <div class="image-card">
                <img src="plots/convergence_{algorithm_name.lower()}.png" alt="Convergence">
                <p>Convergence curve</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Metrics Comparison</h2>
        <div class="image-grid">
            <div class="image-card">
                <img src="plots/metrics_comparison.png" alt="Metrics">
                <p>Metrics comparison</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Reconstruction Examples ({image_count} images)</h2>
        <div class="image-grid">
            <div class="image-card">
                <img src="plots/reconstruction_example.png" alt="Reconstruction">
                <p>Sample reconstruction</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Reports</h2>
        <ul>
            <li><a href="report.json">JSON Report</a></li>
            <li><a href="report.md">Markdown Report</a></li>
            <li><a href="summary.txt">Text Summary</a></li>
        </ul>
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    return index_path