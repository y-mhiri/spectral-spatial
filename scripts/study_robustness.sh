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

SCRIPT_NAME=$(basename "$0")   

cp $SCRIPT_DIR/$SCRIPT_NAME $RESULTS_DIR/$SCRIPT_NAME


echo "Starting robustness study"
echo "Dataset: $DATASET_PATH"
echo "Device:  $DEVICE"
echo "Results: $RESULTS_DIR"
echo ""

run_experiment() {
    local algorithm=$1
    local noise_level=$2
    local sigma_blur=$3
    local softness=$4
    local output_dir="$RESULTS_DIR/${algorithm}_noise${noise_level}_blur${sigma_blur}_softness${softness}"
    local session="${algorithm}_noise${noise_level}_blur${sigma_blur}_softness${softness}"
    local logfile="$output_dir/run.log"

    mkdir -p "$output_dir"
    echo "  Launching [$session]  log -> $logfile"

    screen -dmS "$session" bash -c "
        source '$ENV_FILE' &&
        python experiments/experiment_simple.py \
            --algorithm     '$algorithm' \
            --dataset_path  '$DATASET_PATH' \
            --storage_path  '$output_dir' \
            --lmbda         0.001 \
            --lmbda_m       0.0 \
            --p 2.0 --q 2.0 --r 1.0 \
            --max_iter      5000 \
            --max_iter_cp   10 \
            --tol           1e-8 \
            --noise_level   '$noise_level' \
            --sigma_blur    '$sigma_blur' \
            --threshold_softness '$softness' \
            --scale         4 \
            --device        '$DEVICE' \
        > '$logfile' 2>&1
    "
}


for blur in 8.0; do
    for noise_db in 37; do
        run_experiment "CTV"       "$noise_db" "$blur" "1.0e-5"
        for softness in 1.0e-5 5.0e-5 1.0e-4; do
            run_experiment "GradAlign" "$noise_db" "$blur" "$softness"
        done
    done
done

echo ""
echo "All sessions launched. Monitor with:  screen -ls"
echo "Follow a run with:  tail -f $RESULTS_DIR/<run>/run.log"
echo ""
echo "Analyze with:"
echo "  python analysis/analyze.py --study_dir $RESULTS_DIR --group_by noise_level"
