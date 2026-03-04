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
    local max_iter=$3
    local max_iter_cp=$4
    local output_dir="$RESULTS_DIR/${algorithm}_lambda${lmbda}_iter${max_iter}_itercp${max_iter_cp}"
    
    echo "Running $algorithm with λ=$lmbda..."
    
    python experiments/experiment_simple.py \
        --algorithm "$algorithm" \
        --dataset_path "$DATASET_PATH" \
        --storage_path "$output_dir" \
        --lmbda "$lmbda" \
        --lmbda_m 5.0 \
        --p 2.0 --q 2.0 --r 1.0 \
        --max_iter "${max_iter}" \
        --max_iter_cp ${max_iter_cp} \
        --noise_level 40 \
        --sigma_blur 1.0 \
        --scale 4 \
        --device cuda
    
    if [ $? -eq 0 ]; then
        echo "✓ $algorithm with λ=$lmbda completed successfully"
        
        # Generate visualization
        python analysis/visualization.py \
            --zarr_path "$output_dir/results.zarr" \
            --output_dir "$output_dir/visualization" \
            --rgb_indices 20 10 5 \
            --dataset_path "$DATASET_PATH"
        
        echo "✓ Visualization for $algorithm with λ=$lmbda completed"
    else
        echo "✗ $algorithm with λ=$lmbda failed"
    fi
    echo ""
}

# Study convergence for different regularization weights
# Log scale from 1e-5 to 1e-2 as requested
for lambda in 0.00001 0.0001 0.001 0.01; do
    for it in 10 50 100; do
        run_experiment "CTV" "$lambda" "100" "$it"
        run_experiment "GradAlign" "$lambda" "100" "$it"
    done
done

echo "Convergence study completed!"
echo ""
echo "Results location: $RESULTS_DIR"
echo ""
echo "To generate comprehensive analysis figures, run:"
echo "  python analysis/pipelines/convergence_analysis.py \\"
echo "    --study_dir $RESULTS_DIR \\"
echo "    --output_dir $RESULTS_DIR/analysis"
echo ""