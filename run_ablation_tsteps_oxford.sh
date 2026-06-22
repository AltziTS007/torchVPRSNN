#!/usr/bin/env bash
set -u

echo "============================================================"
echo "    Latency vs. Accuracy Ablation Study (t_steps) - OXFORD  "
echo "============================================================"

# Array of t_steps to test
T_STEPS=(20 50 100 150 200 300 400 500 750 1000)
MAX_SAMPLES=100

mkdir -p tsteps_ablation_results_oxford

for ts in "${T_STEPS[@]}"; do
    echo "Running with t_steps = ${ts} in background..."
    LOG_FILE="tsteps_ablation_results_oxford/tsteps_${ts}.log"
    # Run the model (saving results to terminal/file without writing to the large results dir)
    python snn_model.py --dataset oxford --max-samples $MAX_SAMPLES --t-steps $ts --save-results false > "$LOG_FILE" 2>&1 &
done
wait

echo "Done running ablations. Generating plots..."
python plot_tsteps_ablation_oxford.py
python plot_tsteps_ablation_gradient.py
