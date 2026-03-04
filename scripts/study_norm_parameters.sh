#!/bin/bash
# CTV Norm Parameter Study
# Tests different compound norm combinations (p, q, r) for the CTV algorithm

# Check if DATASET_PATH is set
if [ -z "$DATASET_PATH" ]; then
    echo "ERROR: DATASET_PATH environment variable not set"
    echo "Please set DATASET_PATH to your dataset location"
    exit 1
fi

# Create results directory
RESULTS_DIR="results/norm_study_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Starting CTV Norm Parameter Study"
echo "Dataset: $DATASET_PATH"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run norm parameter test
run_norm_test() {
    local p=$1
    local q=$2
    local r=$3
    local norm_name="p${p}_q${q}_r${r}"
    local output_dir="$RESULTS_DIR/CTV_${norm_name}"
    
    echo "Testing CTV with norm (p=$p, q=$q, r=$r)..."
    
    python experiments/experiment_simple.py \
        --algorithm "CTV" \
        --dataset_path "$DATASET_PATH" \
        --storage_path "$output_dir" \
        --lmbda 0.1 \
        --lmbda_m 1.0 \
        --p "$p" --q "$q" --r "$r" \
        --max_iter 50 \
        --tol 1e-8 \
        --noise_level 0.01 \
        --sigma_blur 1.0 \
        --scale 4
    
    if [ $? -eq 0 ]; then
        echo "✓ Norm (p=$p, q=$q, r=$r) completed"
        
        # Generate visualization
        python analysis/visualization.py \
            --zarr_path "$output_dir/results.zarr" \
            --output_dir "$output_dir/visualization" \
            --rgb_indices 20 10 5
        
        echo "✓ Visualization completed"
    else
        echo "✗ Norm (p=$p, q=$q, r=$r) failed"
    fi
    echo ""
}

echo "Testing scientifically motivated norm combinations..."
echo ""

# Test 4 norm combinations with clear scientific rationale

# 1. L2,1 norm: Group sparsity across gradients (spectral preservation)
run_norm_test 2.0 1.0 1.0

# 2. L2,2 norm: Isotropic TV (standard baseline)
run_norm_test 2.0 2.0 1.0

# 3. L1,1 norm: Anisotropic TV (aggressive sparsity)
run_norm_test 1.0 1.0 1.0

# 4. L∞,1 norm: Max norm across bands (robust to outliers)
run_norm_test inf 1.0 1.0

echo "Norm parameter study completed!"
echo ""
echo "Summary of results:"
echo "- 4 norm combinations tested"
echo "- Fixed parameters: λ=0.1, λ_m=1.0, noise=0.01, blur=1.0"
echo "- Results location: $RESULTS_DIR"
echo ""

echo "Norm combination analysis:"
echo "================================"
echo "1. L2,1 (p=2,q=1,r=1): Group sparsity - expected to preserve spectral signatures"
echo "2. L2,2 (p=2,q=2,r=1): Isotropic TV - standard baseline for comparison"
echo "3. L1,1 (p=1,q=1,r=1): Anisotropic TV - more aggressive edge preservation"
echo "4. L∞,1 (p=∞,q=1,r=1): Max norm - robust to outliers and strong gradients"
echo ""

echo "To compare results:"
echo "cd $RESULTS_DIR"
echo "ls -la */visualization/metrics_*.png | xargs ls -lh"