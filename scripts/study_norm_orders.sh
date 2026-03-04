#!/bin/bash
# Norm Order Study Script
# Compares different norm combinations (p, q, r) for both algorithms

# Check if DATASET_PATH is set
if [ -z "$DATASET_PATH" ]; then
    echo "ERROR: DATASET_PATH environment variable not set"
    echo "Please set DATASET_PATH to your dataset location"
    exit 1
fi

# Create results directory
RESULTS_DIR="results/norm_order_study_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Starting Norm Order Study"
echo "Dataset: $DATASET_PATH"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run single experiment
run_experiment() {
    local algorithm=$1
    local p=$2
    local q=$3
    local r=$4
    local test_name="${algorithm}_p${p}_q${q}_r${r}"
    local output_dir="$RESULTS_DIR/$test_name"
    
    echo "Running $algorithm with p=$p, q=$q, r=$r..."
    
    python experiments/experiment_simple.py \
        --algorithm "$algorithm" \
        --dataset_path "$DATASET_PATH" \
        --storage_path "$output_dir" \
        --lmbda 0.01 \
        --lmbda_m 1.0 \
        --p $p --q $q --r $r \
        --max_iter 50 \
        --tol 1e-8 \
        --noise_level 40 \
        --sigma_blur 1.0 \
        --scale 4 \
        --device cuda
    
    if [ $? -eq 0 ]; then
        echo "✓ $algorithm with p=$p, q=$q, r=$r completed successfully"
        
        # Generate visualization
        python analysis/visualization.py \
            --zarr_path "$output_dir/results.zarr" \
            --output_dir "$output_dir/visualization" \
            --rgb_indices 20 10 5
        
        echo "✓ Visualization completed"
    else
        echo "✗ $algorithm with p=$p, q=$q, r=$r failed"
    fi
    echo ""
}

# Norm order combinations to test
# Format: p, q, r
norm_combinations=(
    "2 2 1"   # 221 - Common choice
    "1 1 1"   # 111 - L1 norm
    "1 1 inf" # 11inf - Mixed norms
    "2 1 1"   # 211 - Alternative
    "1 2 1"   # 121 - Alternative
)

# Study norm orders for both algorithms
echo "Testing norm order combinations:"
echo "="
for combo in "${norm_combinations[@]}"; do
    read p q r <<< "$combo"
    echo "  p=$p, q=$q, r=$r"
done
echo "="
echo ""

for combo in "${norm_combinations[@]}"; do
    read p q r <<< "$combo"
    
    # Run for both algorithms
    run_experiment "CTV" $p $q $r
    run_experiment "GradAlign" $p $q $r
    
    echo "----------------------------------------"
done

echo "Norm order study completed!"
echo ""
echo "Results location: $RESULTS_DIR"
echo ""
echo "Summary of norm combinations tested:"
echo "  1. 2-2-1 (standard)"
echo "  2. 1-1-1 (L1)"
echo "  3. 1-1-inf (mixed)"
echo "  4. 2-1-1 (alternative)"
echo "  5. 1-2-1 (alternative)"
echo ""
echo "To analyze results:"
echo "  python analysis/pipelines/norm_order_comparison.py \\"
echo "    --study_dir $RESULTS_DIR \\"
echo "    --output_dir $RESULTS_DIR/analysis"
