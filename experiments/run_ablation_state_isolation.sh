#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "      State Isolation (Temporal Spill-over) Ablation        "
echo "============================================================"
echo "This script runs the SNN model with its core learning mechanisms"
echo "(Hard WTA, local homeostasis, patch normalization) ENABLED."
echo ""
echo "It compares two conditions:"
echo "It compares two conditions on a HEALTHY network:"
echo "1. State Isolation ON (Reset): The network state is reset between images."
echo "2. State Isolation OFF (Spill-over): Residual state carries over to the next image."
echo "============================================================"

# Default to 100 samples if not provided, for faster ablation testing
MAX_SAMPLES=${1:-100}

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/ablation_logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/state_isolation_ablation_${TIMESTAMP}.log"

echo "Starting Ablation Study at $(date)" | tee -a "$LOG_FILE"
echo "MAX_SAMPLES: $MAX_SAMPLES" | tee -a "$LOG_FILE"
echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo ">>> RUN 1: Healthy Network + State Isolation ON (Reset) <<<" | tee -a "$LOG_FILE"
echo "Command: python "$ROOT_DIR"/core/snn_model.py --max-samples $MAX_SAMPLES --save-results true --experiment-name healthy_reset" | tee -a "$LOG_FILE"
python "$ROOT_DIR"/core/snn_model.py \
    --max-samples "$MAX_SAMPLES" \
    --save-results true \
    --experiment-name healthy_reset \
    | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo ">>> RUN 2: Healthy Network + State Isolation OFF (Spill-over) <<<" | tee -a "$LOG_FILE"
echo "Command: python "$ROOT_DIR"/core/snn_model.py --max-samples $MAX_SAMPLES --simulate-spillover --save-results true --experiment-name healthy_spillover" | tee -a "$LOG_FILE"
python "$ROOT_DIR"/core/snn_model.py \
    --max-samples "$MAX_SAMPLES" \
    --simulate-spillover \
    --save-results true \
    --experiment-name healthy_spillover \
    | tee -a "$LOG_FILE"

echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
echo "Ablation Study completed at $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: $LOG_FILE"
