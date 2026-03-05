#!/bin/bash
# Norm order study: compare (p, q, r) norm combinations for both algorithms.
# Answers: which l^{p,q,r} norm gives the best reconstruction?
#
# Usage: DATASET_PATH=/path/to/data.zarr [DEVICE=cpu] ./scripts/study_norm.sh

if [ -z "$DATASET_PATH" ]; then
    echo "ERROR: DATASET_PATH environment variable not set"
    exit 1
fi

DEVICE=${DEVICE:-cuda}
RESULTS_DIR="results/norm_study_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Starting norm order study"
echo "Dataset: $DATASET_PATH"
echo "Device:  $DEVICE"
echo "Results: $RESULTS_DIR"
echo ""

run_experiment() {
    local algorithm=$1
    local p=$2
    local q=$3
    local r=$4
    local output_dir="$RESULTS_DIR/${algorithm}_p${p}_q${q}_r${r}"

    echo "Running $algorithm  p=$p  q=$q  r=$r ..."

    python experiments/experiment_simple.py \
        --algorithm     "$algorithm" \
        --dataset_path  "$DATASET_PATH" \
        --storage_path  "$output_dir" \
        --lmbda         0.1 \
        --lmbda_m       1.0 \
        --p "$p" --q "$q" --r "$r" \
        --max_iter      50 \
        --max_iter_cp   50 \
        --tol           1e-8 \
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

# Norm combinations to test: "p q r"
norm_combinations=(
    "2 2 1"    # standard isotropic TV
    "1 1 1"    # L1 (sparsity-promoting)
    "1 1 inf"  # mixed L1/Linf
    "2 1 1"    # alternative
    "1 2 1"    # alternative
)

for combo in "${norm_combinations[@]}"; do
    read p q r <<< "$combo"
    run_experiment "CTV"       "$p" "$q" "$r"
    run_experiment "GradAlign" "$p" "$q" "$r"
done

echo "Done. Results: $RESULTS_DIR"
echo ""
echo "Analyze with:"
echo "  python analysis/analyze.py --study_dir $RESULTS_DIR --group_by p"
