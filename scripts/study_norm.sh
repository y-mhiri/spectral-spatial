#!/bin/bash
# Norm order study: compare (p, q, r) norm combinations for both algorithms.
# Answers: which l^{p,q,r} norm gives the best reconstruction?
#
# Usage: DATASET_PATH=/path/to/data.zarr [DEVICE=cpu] ./scripts/study_norm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/env.sh"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Copy scripts/env.sh.example to scripts/env.sh and fill in your paths."
    exit 1
fi

source "$ENV_FILE"
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
    local session="${algorithm}_p${p}_q${q}_r${r}"
    local logfile="$output_dir/run.log"

    mkdir -p "$output_dir"
    echo "  Launching [$session]  log -> $logfile"

    screen -dmS "$session" bash -c "
        source '$ENV_FILE' &&
        python experiments/experiment_simple.py \
            --algorithm     '$algorithm' \
            --dataset_path  '$DATASET_PATH' \
            --storage_path  '$output_dir' \
            --lmbda         0.1 \
            --lmbda_m       1.0 \
            --p '$p' --q '$q' --r '$r' \
            --max_iter      50 \
            --max_iter_cp   50 \
            --tol           1e-8 \
            --noise_level   40 \
            --sigma_blur    1.0 \
            --scale         4 \
            --device        '$DEVICE' \
        > '$logfile' 2>&1
    "
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

echo ""
echo "All sessions launched. Monitor with:  screen -ls"
echo "Follow a run with:  tail -f $RESULTS_DIR/<run>/run.log"
echo ""
echo "Analyze with:"
echo "  python analysis/analyze.py --study_dir $RESULTS_DIR --group_by p"
