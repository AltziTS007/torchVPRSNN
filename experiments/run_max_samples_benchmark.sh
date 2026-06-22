#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Run snn_model.py 20 times for each requested MAX_SAMPLES value.
# Usage examples:
#   ./run_max_samples_benchmark.sh
#   ./run_max_samples_benchmark.sh --device cuda --save-results false

REPEATS=15
MAX_SAMPLES_LIST=(25 50 100 150 200 250 300 350 400)
SCRIPT_PATH="$ROOT_DIR/core/snn_model.py"
SCRIPT_ARGS="--dataset nordland"

DEVICE="auto"
SAVE_RESULTS="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --save-results)
      SAVE_RESULTS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--device cpu|cuda|auto] [--save-results true|false]"
      exit 1
      ;;
  esac
done

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Error: $SCRIPT_PATH not found in current directory: $PWD"
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT_DIR/benchmark_log"
LOG_FILE="$ROOT_DIR/benchmark_log/benchmark_max_samples_${TIMESTAMP}.log"

echo "Starting benchmark at $(date)" | tee -a "$LOG_FILE"
echo "Device: $DEVICE" | tee -a "$LOG_FILE"
echo "Save results: $SAVE_RESULTS" | tee -a "$LOG_FILE"
echo "Repeats per setup: $REPEATS" | tee -a "$LOG_FILE"
echo "MAX_SAMPLES values: ${MAX_SAMPLES_LIST[*]}" | tee -a "$LOG_FILE"
echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

for max_samples in "${MAX_SAMPLES_LIST[@]}"; do
  for run_idx in $(seq 1 "$REPEATS"); do
    start_epoch_s="$(date +%s)"

    echo "[$(date '+%F %T')] MAX_SAMPLES=$max_samples run=$run_idx/$REPEATS START" | tee -a "$LOG_FILE"

    python "$SCRIPT_PATH" $SCRIPT_ARGS \
      --device "$DEVICE" \
      --save-results "$SAVE_RESULTS" \
      --max-samples "$max_samples" \
      --seed "$((run_idx + 15))"

    exit_code=$?

    end_epoch_s="$(date +%s)"
    elapsed_s=$((end_epoch_s - start_epoch_s))

    if [[ $exit_code -eq 0 ]]; then
      status="OK"
    else
      status="FAIL"
    fi

    echo "[$(date '+%F %T')] MAX_SAMPLES=$max_samples run=$run_idx/$REPEATS END status=$status exit_code=$exit_code elapsed_s=${elapsed_s}" | tee -a "$LOG_FILE"

    if [[ $exit_code -ne 0 ]]; then
      echo "Stopping early due to failure." | tee -a "$LOG_FILE"
      exit $exit_code
    fi
  done
done

echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
echo "Benchmark completed at $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
