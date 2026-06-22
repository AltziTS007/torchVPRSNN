#!/usr/bin/env bash
set -u

# Array of chunk sizes (N places) to test
CHUNK_SIZES=(25 50 100 150 200 250 300 350 400)
TOTAL_PLACES=400

echo "============================================================"
echo "    Sliding Window Protocol - OXFORD FULL CURVE    "
echo "============================================================"

mkdir -p sliding_window_full_results_oxford
rm -f sliding_window_full_results_oxford/summary_*.txt

for CS in "${CHUNK_SIZES[@]}"; do
    echo "Running Sliding Window Protocol for Chunk Size: $CS"
    
    if [ "$CS" -eq "$TOTAL_PLACES" ]; then
        NUM_CHUNKS=1
        STRIDE=0
    else
        # Ceiling division for minimum number of chunks to cover the dataset
        NUM_CHUNKS=$(( (TOTAL_PLACES + CS - 1) / CS ))
        
        # Enforce the user's rule: at least two sections with varying offsets
        if [ "$NUM_CHUNKS" -lt 2 ]; then
            NUM_CHUNKS=2
        fi
        
        # Calculate exactly how far to slide the window each time to space them out evenly
        STRIDE=$(( (TOTAL_PLACES - CS) / (NUM_CHUNKS - 1) ))
    fi
    
    echo "Number of runs for CS=$CS: $NUM_CHUNKS (Stride: $STRIDE)"
    
    OUT_DIR="sliding_window_full_results_oxford/cs_${CS}"
    mkdir -p "$OUT_DIR"
    
    for (( i=0; i<NUM_CHUNKS; i++ )); do
        START_IDX=$(( i * STRIDE ))
        LOG_FILE="${OUT_DIR}/chunk_${i}.log"
        echo "  -> Running chunk $i (Start: $START_IDX, Size: $CS) in background..."
        python snn_model.py --dataset oxford --start-idx $START_IDX --max-samples $CS --t-steps 150 --save-results false > "$LOG_FILE" 2>&1 &
    done
    wait
    
    # Parse the results for this chunk size
    python plot_sliding_window_protocol.py --dir "$OUT_DIR" > "sliding_window_full_results_oxford/summary_${CS}.txt"
    cat "sliding_window_full_results_oxford/summary_${CS}.txt"
    echo "------------------------------------------------------------"
done

echo "Done running all chunk sizes. Plotting final curve..."
python plot_sliding_window_curve_oxford.py
