#!/bin/bash
# Convergence study: sweep λ and Chambolle-Pock iterations for both algorithms.
# Answers: does the algorithm converge, and how fast, as a function of these two params?
#
# Usage: DATASET_PATH=/path/to/data.zarr [DEVICE=cpu] ./scripts/study_convergence.sh

if [ -z "$DATASET_PATH" ]; then
    echo "ERROR: DATASET_PATH environment variable not set"
    exit 1
fi

DEVICE=${DEVICE:-cuda}
RESULTS_DIR="results/convergence_study_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Starting convergence study"
echo "Dataset: $DATASET_PATH"
echo "Device:  $DEVICE"
echo "Results: $RESULTS_DIR"
echo ""

run_experiment() {
    local algorithm=$1
    local lmbda=$2
    local max_iter_cp=$3
    local output_dir="$RESULTS_DIR/${algorithm}_lambda${lmbda}_cp${max_iter_cp}"

    echo "Running $algorithm  λ=$lmbda  CP_iter=$max_iter_cp ..."

    python experiments/experiment_simple.py \
        --algorithm     "$algorithm" \
        --dataset_path  "$DATASET_PATH" \
        --storage_path  "$output_dir" \
        --lmbda         "$lmbda" \
        --lmbda_m       1.0 \
        --p 2.0 --q 2.0 --r 1.0 \
        --max_iter      100 \
        --max_iter_cp   "$max_iter_cp" \
        --noise_level   40 \
        --sigma_blur    1.0 \
        --scale         4 \
        --device        "$DEVICE"

    if [ $? -eq 0 ]; then
        echo "  ✓ done"
    else
        echo "  ✗ failed"
    fi
    echo ""
}

for lmbda in 0.00001 0.0001 0.001 0.01; do
    for cp_iter in 10 50 100; do
        run_experiment "CTV"       "$lmbda" "$cp_iter"
        run_experiment "GradAlign" "$lmbda" "$cp_iter"
    done
done

echo "Done. Results: $RESULTS_DIR"
echo ""
echo "Analyze with:"
echo "  python analysis/analyze.py --study_dir $RESULTS_DIR --group_by lmbda"
