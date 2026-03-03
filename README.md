# Minimal Hyperspectral Pansharpening Experiments

A streamlined codebase for hyperspectral image pansharpening experiments using TV-based regularization.

## Quick Start

```bash
# Set your dataset path
export DATASET_PATH="/path/to/your/dataset.zarr"

# Run convergence study (8 experiments)
./scripts/study_convergence.sh

# Run performance study (8 experiments)
./scripts/study_performance.sh

# Or run individual experiments
python experiments/experiment_simple.py \
  --algorithm CTV \
  --dataset_path "$DATASET_PATH" \
  --storage_path results/my_experiment
```

## Structure

```
src/
├── algorithms/       # Core optimization algorithms
├── datasets/         # Data loading and simulation
└── metrics/          # Evaluation metrics

experiments/          # Experiment runners and configurations
analysis/             # Visualization and result analysis
```

## Key Features

- **Two algorithms**: CTV and CTV with gradient alignment
- **Simplified interface**: Sensible defaults, clear documentation
- **Standardized visualization**: Publication-quality plots
- **Minimal dependencies**: Only essential components
- **Comprehensive scripts**: Ready-to-use experiment launchers

## Experiment Scripts

The `scripts/` directory contains pre-configured bash scripts for common experimental workflows:

- **`study_convergence.sh`**: Tests algorithm convergence with different regularization weights
- **`study_performance.sh`**: Compares performance under various degradation conditions

Both scripts automatically:
- Create timestamped result directories
- Generate comprehensive visualizations
- Save all metrics and figures
- Provide progress updates

See `scripts/README.md` for detailed usage.

## Requirements

See `requirements.txt` for dependencies.

## Data

Place your hyperspectral datasets in the `data/` directory. The code expects Zarr format with appropriate metadata.