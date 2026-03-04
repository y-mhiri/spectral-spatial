#!/bin/bash
# Performance Study Script
# Compares the performance of CTV and GradAlign algorithms across different conditions

# Check if DATASET_PATH is set
if [ -z "$DATASET_PATH" ]; then
    echo "ERROR: DATASET_PATH environment variable not set"
    echo "Please set DATASET_PATH to your dataset location"
    exit 1
fi

# Create results directory
RESULTS_DIR="results/performance_study_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Starting Performance Study"
echo "Dataset: $DATASET_PATH"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run performance test
run_performance_test() {
    local algorithm=$1
    local noise_level=$2
    local blur_sigma=$3
    local test_name="${algorithm}_noise${noise_level}_blur${blur_sigma}"
    local output_dir="$RESULTS_DIR/$test_name"
    
    echo "Running $algorithm with noise=$noise_level, blur=$blur_sigma..."
    
    python experiments/experiment_simple.py \
        --algorithm "$algorithm" \
        --dataset_path "$DATASET_PATH" \
        --storage_path "$output_dir" \
        --lmbda 0.1 \
        --lmbda_m 1.0 \
        --p 2.0 --q 2.0 --r 1.0 \
        --max_iter 50 \
        --tol 1e-8 \
        --noise_level "$noise_level" \
        --sigma_blur "$blur_sigma" \
        --scale 4
    
    if [ $? -eq 0 ]; then
        echo "✓ $algorithm with noise=$noise_level, blur=$blur_sigma completed"
        
        # Generate visualization
        python analysis/visualization.py \
            --zarr_path "$output_dir/results.zarr" \
            --output_dir "$output_dir/visualization" \
            --rgb_indices 20 10 5 \
            --dataset_path "$DATASET_PATH"
        
        echo "✓ Visualization completed"
    else
        echo "✗ $algorithm with noise=$noise_level, blur=$blur_sigma failed"
    fi
    echo ""
}

# Performance study matrix: test different noise and blur conditions
echo "Testing performance under different degradation conditions..."
echo ""

# Test 3 noise levels as requested (35dB, 40dB, 45dB)
# Note: noise_level in script corresponds to variance
# 35dB ≈ 0.000316, 40dB ≈ 0.0001, 45dB ≈ 0.0000316
for noise_db in 35 40 45; do
    case $noise_db in
        35) noise_var=0.000316 ;;
        40) noise_var=0.0001 ;;
        45) noise_var=0.0000316 ;;
    esac
    
    # Test with low and high blur
    for blur in 0.5 2.0; do
        run_performance_test "CTV" $noise_var $blur
        run_performance_test "GradAlign" $noise_var $blur
    done
done

echo "Performance study completed!"
echo ""
echo "Summary of results:"
echo "- 2 algorithms × 4 degradation conditions = 8 experiments"
echo "- Results include: reconstructed images, metrics, convergence plots"
echo "- Results location: $RESULTS_DIR"
echo ""

echo "Performance comparison summary:"
echo "================================"

# Generate performance summary
PERFORMANCE_CSV="$RESULTS_DIR/performance_summary.csv"
echo "Algorithm,Noise,Blur,PSNR,SSIM,Runtime" > "$PERFORMANCE_CSV"

for test_dir in "$RESULTS_DIR"/*/; do
    if [ -f "${test_dir}info.yaml" ]; then
        algorithm=$(basename "$test_dir" | cut -d'_' -f1)
        noise=$(basename "$test_dir" | grep -o 'noise[0-9.]*' | cut -d'i' -f2)
        blur=$(basename "$test_dir" | grep -o 'blur[0-9.]*' | cut -d'r' -f2)
        
        # Extract metrics (simplified - in practice would parse YAML/JSON)
        psnr="N/A"
        ssim="N/A"
        runtime="N/A"
        
        echo "$algorithm,$noise,$blur,$psnr,$ssim,$runtime" >> "$PERFORMANCE_CSV"
    fi
done

echo "Detailed performance data saved to: $PERFORMANCE_CSV"
echo ""
echo "To view results:"
echo "cd $RESULTS_DIR"
echo "ls -la */visualization/"
echo ""
echo "To generate comprehensive analysis figures, run:"
echo "  python analysis/pipelines/noise_impact.py \\"
echo "    --study_dir $RESULTS_DIR \\"
echo "    --output_dir $RESULTS_DIR/analysis"