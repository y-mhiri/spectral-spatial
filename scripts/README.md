# Experiment Launch Scripts

This directory contains bash scripts to launch comprehensive experiments for hyperspectral pansharpening algorithms.

## Prerequisites

Set the `DATASET_PATH` environment variable to your dataset location:

```bash
export DATASET_PATH="/path/to/your/dataset.zarr"
```

## Available Scripts

### 1. Convergence Study

**Script**: `study_convergence.sh`

**Purpose**: Studies the convergence behavior of both CTV and GradAlign algorithms with different regularization weights.

**Parameters Tested**:
- Algorithms: CTV, GradAlign
- Regularization weights (λ): 0.01, 0.05, 0.1, 0.2
- Total experiments: 8

**Usage**:
```bash
./study_convergence.sh
```

**Output**:
- Convergence plots for each parameter setting
- Reconstructed images and error maps
- Performance metrics (PSNR, SSIM, etc.)
- Organized results in `results/convergence_study_[timestamp]/`

### 2. Performance Study

**Script**: `study_performance.sh`

**Purpose**: Compares algorithm performance under different degradation conditions.

**Conditions Tested**:
- Noise levels: 0.001 (low), 0.05 (high)
- Blur levels: 0.5 (low), 2.0 (high)
- Total experiments: 8 (2 algorithms × 4 conditions)

**Usage**:
```bash
./study_performance.sh
```

**Output**:
- Performance metrics for each condition
- Visualization of results
- CSV summary of all experiments
- Organized results in `results/performance_study_[timestamp]/`

## Results Organization

Both scripts create timestamped result directories with this structure:

```
results/[study_type]_[timestamp]/
├── [algorithm]_[parameters]/
│   ├── results.zarr          # Raw experiment data
│   ├── info.yaml             # Experiment metadata
│   └── visualization/
│       ├── convergence_*.png # Convergence plots
│       ├── metrics_*.png      # Metrics comparison
│       └── reconstruction_*.png # Image examples
└── performance_summary.csv  # Aggregated results (performance study only)
```

## Customization

To modify experiment parameters:

1. **Convergence Study**: Edit `study_convergence.sh` and change the `lambda` values in the for loop.

2. **Performance Study**: Edit `study_performance.sh` and modify the test conditions.

## Tips

- **Monitor progress**: Each script provides real-time progress updates
- **Error handling**: Scripts check for `DATASET_PATH` and report errors clearly
- **Parallel execution**: For faster results, you can run multiple scripts simultaneously
- **Resource management**: The scripts automatically clean up PyTorch cache between runs

## Example Workflow

```bash
# Set dataset path
export DATASET_PATH="/data/harvard.zarr"

# Run convergence study
./study_convergence.sh

# Run performance study  
./study_performance.sh

# Analyze results
cd results
find . -name "*.png" | xargs ls -lh  # List all generated figures
```

## Notes

- All scripts assume the dataset is in Zarr format
- RGB visualization uses bands 20, 10, 5 by default (adjust in scripts if needed)
- Results are saved with timestamps to avoid overwriting
- Each experiment runs with sensible default parameters optimized for typical hyperspectral data