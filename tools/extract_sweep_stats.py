import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

import os
import glob
import numpy as np
import argparse
from datetime import datetime

def extract_sweep_stats():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="Start time e.g., '2026-06-17 14:21:21'")
    parser.add_argument("--end", type=str, help="End time e.g., '2026-06-17 16:30:22'")
    args = parser.parse_args()
    
    start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S") if args.start else datetime.min
    end_dt = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S") if args.end else datetime.max

    # Get all results directories
    all_dirs = glob.glob('results/results_*')
    
    valid_dirs = []
    for d in all_dirs:
        # e.g., results_2026-06-16_14-32-32
        ts_str = d.replace('results/results_', '').replace('-', ':')
        try:
            ts_str_clean = ts_str[:10].replace(':', '-') + " " + ts_str[11:]
            dir_time = datetime.strptime(ts_str_clean, "%Y-%m-%d %H:%M:%S")
            if start_dt <= dir_time <= end_dt:
                valid_dirs.append(d)
        except ValueError:
            pass
            
    # Sort chronologically
    valid_dirs.sort()
    # let's just parse them and filter for the Nordland 100-sample STDP runs.
    
    k_scores = {}
    
    run_count = 0
    for rdir in valid_dirs:
        log_file = os.path.join(rdir, 'output.log')
        if not os.path.exists(log_file):
            continue
            
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        # Verify it's a 100-sample run
        if not any("Limiting dataset to 100 samples" in l for l in lines):
            continue
            
        # Parse the sequential aggregation section
        in_sweep_section = False
        in_product_rule = False
        
        current_run_scores = {}
        
        for line in lines:
            if "--- Running Sequential Frame Aggregation Sweep ---" in line:
                in_sweep_section = True
            elif in_sweep_section and "Aggregation Rule: product" in line:
                in_product_rule = True
            elif in_sweep_section and "Aggregation Rule: mean" in line:
                in_product_rule = False
            elif in_sweep_section and "Processing Results for" in line:
                in_sweep_section = False
                
            if in_sweep_section and in_product_rule:
                if "k=" in line and "R@100P =" in line:
                    # Example: k= 5 (N=96): R@100P = 100.00% | P@100R = 100.00% | Acc = 100.00%
                    parts = line.split(":")
                    k_part = parts[0].strip() # "k= 5 (N=96)"
                    k_val = int(k_part.split("k=")[1].split("(")[0].strip())
                    
                    r100p_part = parts[1].split("|")[0].strip() # "R@100P = 100.00%"
                    r100p_val = float(r100p_part.split("=")[1].replace("%", "").strip())
                    
                    current_run_scores[k_val] = r100p_val
                    
        if current_run_scores:
            run_count += 1
            for k, score in current_run_scores.items():
                if k not in k_scores:
                    k_scores[k] = []
                k_scores[k].append(score)
                
            # If we hit 15 valid runs, stop
            if run_count == 15:
                break
                
    if run_count == 0:
        print("No valid runs found matching the criteria.")
        return
        
    print(f"\nExtracted Sequential Aggregation (Product Rule) across {run_count} runs:\n")
    print(f"{'k (Window Size)':<20} | {'Mean R@100P':<15} | {'Std Dev':<15}")
    print("-" * 55)
    
    for k in sorted(k_scores.keys()):
        scores = k_scores[k]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"k = {k:<16} | {mean_score:>6.2f}%         | ±{std_score:>6.2f}%")
        
    print("\n")
    
if __name__ == "__main__":
    extract_sweep_stats()
