#!/bin/bash
# Convergence study: sweep λ and Chambolle-Pock iterations for both algorithms.
# Answers: does the algorithm converge, and how fast, as a function of these two params?
#
# Usage: DATASET_PATH=/path/to/data.zarr [DEVICE=cpu] ./scripts/study_convergence.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/env.sh"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Copy scripts/env.sh.example to scripts/env.sh and fill in your paths."
    exit 1
fi

# Allow overriding DATASET_PATH and DEVICE from the shell; env.sh sets defaults.
source "$ENV_FILE"
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
    local session="${algorithm}_l${lmbda}_cp${max_iter_cp}"
    local logfile="$output_dir/run.log"

    mkdir -p "$output_dir"
    echo "  Launching [$session]  log -> $logfile"

    screen -dmS "$session" bash -c "
        source '$ENV_FILE' &&
        python experiments/experiment_simple.py \
            --algorithm     '$algorithm' \
            --dataset_path  '$DATASET_PATH' \
            --storage_path  '$output_dir' \
            --lmbda         '$lmbda' \
            --lmbda_m       1.0 \
            --p 2.0 --q 2.0 --r 1.0 \
            --max_iter      10000 \
            --max_iter_cp   '$max_iter_cp' \
            --noise_level   45 \
            --sigma_blur    1.0 \
            --scale         4 \
            --device        '$DEVICE' \
        > '$logfile' 2>&1
    "
}

for lmbda in 0.00001 0.0001 0.001 0.01; do
    for cp_iter in 10 50 100; do
        run_experiment "CTV"       "$lmbda" "$cp_iter"
        run_experiment "GradAlign" "$lmbda" "$cp_iter"
    done
done

echo ""
echo "All sessions launched. Monitor with:  screen -ls"
echo "Follow a run with:  tail -f $RESULTS_DIR/<run>/run.log"
echo ""
echo "Analyze with:"
echo "  python analysis/analyze_convergence.py --study_dir $RESULTS_DIR"
