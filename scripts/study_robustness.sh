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
    local lmbda=$2
    local lmbda_m=$3
    local noise_level=$4
    # local sigma_blur=$5

    local output_dir="$RESULTS_DIR/${algorithm}_lmbda${lmbda}_lmbdaM${lmbda_m}_noise${noise_level}"
    local session="${algorithm}_lmbda${lmbda}_lmbdaM${lmbda_m}_noise${noise_level}"
    local logfile="$output_dir/run.log"

    mkdir -p "$output_dir"
    echo "  Launching [$session]  log -> $logfile"

    screen -dmS "$session" bash -c "
        source '$ENV_FILE' &&
        python experiments/experiment_simple.py \
            --algorithm     '$algorithm' \
            --dataset_path  '$DATASET_PATH' \
            --storage_path  '$output_dir' \
            --lmbda         $lmbda \
            --lmbda_m       $lmbda_m \
            --p 2.0 --q 2.0 --r 1.0 \
            --max_iter      5000 \
            --max_iter_cp   10 \
            --tol           1e-8 \
            --noise_level   $noise_level \
            --sigma_blur    4.0 \
            --threshold_softness 1.0e-5 \
            --scale         4 \
            --device        '$DEVICE' \
        > '$logfile' 2>&1
    "
}

for noise_level in 30 37; do
for lmbda in 0.01 0.1; do
    for lmbda_m in 5.0 8.0; do
        run_experiment "CTV"       "$lmbda" "$lmbda_m" "$noise_level" 
        run_experiment "GradAlign" "$lmbda" "$lmbda_m" "$noise_level" 
    done    
done
done

echo ""
echo "All sessions launched. Monitor with:  screen -ls"
echo "Follow a run with:  tail -f $RESULTS_DIR/<run>/run.log"
echo ""