#!/bin/bash
# Convergence Study Script
# Studies the convergence behavior of both CTV and GradAlign algorithms

# Check if DATASET_PATH is set
if [ -z "$DATASET_PATH" ]; then
    echo "ERROR: DATASET_PATH environment variable not set"
    echo "Please set DATASET_PATH to your dataset location"
    exit 1
fi

# Create results directory
RESULTS_DIR="results/convergence_study_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Starting Convergence Study"
echo "Dataset: $DATASET_PATH"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run single experiment
run_experiment() {
    local algorithm=$1
    local lmbda=$2
    local output_dir="$RESULTS_DIR/${algorithm}_lambda${lmbda}"
    
    echo "Running $algorithm with λ=$lmbda..."
    
    python experiments/experiment_simple.py \
        --algorithm "$algorithm" \
        --dataset_path "$DATASET_PATH" \
        --storage_path "$output_dir" \
        --lmbda "$lmbda" \
        --lmbda_m 1.0 \
        --p 2.0 --q 2.0 --r 1.0 \
        --max_iter 100 \
        --tol 1e-8 \
        --noise_level 0.01 \
        --sigma_blur 1.0 \
        --scale 4
    
    if [ $? -eq 0 ]; then
        echo "✓ $algorithm with λ=$lmbda completed successfully"
        
        # Generate visualization
        python analysis/visualization.py \
            --zarr_path "$output_dir/results.zarr" \
            --output_dir "$output_dir/visualization" \
            --rgb_indices 20 10 5
        
        echo "✓ Visualization for $algorithm with λ=$lmbda completed"
    else
        echo "✗ $algorithm with λ=$lmbda failed"
    fi
    echo ""
}

# Study convergence for different regularization weights
for lambda in 0.01 0.05 0.1 0.2; do
    run_experiment "CTV" "$lambda"
    run_experiment "GradAlign" "$lambda"
done

echo "Convergence study completed!"
echo ""
echo "Summary of results:"
echo "- CTV algorithm: 4 parameter settings tested"
echo "- GradAlign algorithm: 4 parameter settings tested"
echo "- Total experiments: 8"
echo "- Results location: $RESULTS_DIR"
echo ""
echo "To analyze convergence:"
echo "cd $RESULTS_DIR"
echo "find . -name "convergence_*.png" | xargs ls -lh"