#!/bin/bash
# Robustness study: compare CTV and GradAlign across noise and blur conditions.
# Answers: how does reconstruction quality degrade with harder degradation?
#
# Usage: DATASET_PATH=/path/to/data.zarr [DEVICE=cpu] ./scripts/study_robustness.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/env.sh"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Copy scripts/env.sh.example to scripts/env.sh and fill in your paths."
    exit 1
fi

source "$ENV_FILE"
DEVICE=${DEVICE:-cuda}
RESULTS_DIR="results/robustness_study_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Starting robustness study"
echo "Dataset: $DATASET_PATH"
echo "Device:  $DEVICE"
echo "Results: $RESULTS_DIR"
echo ""

run_experiment() {
    local algorithm=$1
    local noise_level=$2
    local sigma_blur=$3
    local output_dir="$RESULTS_DIR/${algorithm}_noise${noise_level}_blur${sigma_blur}"
    local session="${algorithm}_noise${noise_level}_blur${sigma_blur}"
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
            --p 2.0 --q 2.0 --r 1.0 \
            --max_iter      50 \
            --max_iter_cp   50 \
            --tol           1e-8 \
            --noise_level   '$noise_level' \
            --sigma_blur    '$sigma_blur' \
            --scale         4 \
            --device        '$DEVICE' \
        > '$logfile' 2>&1
    "
}

# 3 noise levels × 2 blur levels × 2 algorithms = 12 experiments
for noise_db in 35 40 45; do
    for blur in 0.5 2.0; do
        run_experiment "CTV"       "$noise_db" "$blur"
        run_experiment "GradAlign" "$noise_db" "$blur"
    done
done

echo ""
echo "All sessions launched. Monitor with:  screen -ls"
echo "Follow a run with:  tail -f $RESULTS_DIR/<run>/run.log"
echo ""
echo "Analyze with:"
echo "  python analysis/analyze.py --study_dir $RESULTS_DIR --group_by noise_level"
