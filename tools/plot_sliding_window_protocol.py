import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

import os
import re
import argparse
import numpy as np

def parse_log(filepath):
    r100p = None
    auc_pr = None
    with open(filepath, 'r') as f:
        content = f.read()
    # Format: Weighted+Prob | 94.00% | 94.00% | 90.18% | 92.50%
    match = re.search(r'Weighted\+Prob\s+\|\s*[\d\.]+\s*%\s*\|\s*[\d\.]+\s*%\s*\|\s*([\d\.]+)\s*%\s*\|\s*([\d\.]+)\s*%', content)
    if match:
        r100p = float(match.group(1))
        auc_pr = float(match.group(2))
    return r100p, auc_pr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="sliding_window_ablation_results")
    args = parser.parse_args()

    results_r100p = []
    results_auc = []
    
    for filename in os.listdir(args.dir):
        if filename.startswith("chunk_") and filename.endswith(".log"):
            filepath = os.path.join(args.dir, filename)
            r100p, auc_pr = parse_log(filepath)
            if r100p is not None:
                results_r100p.append(r100p)
                results_auc.append(auc_pr)

    if not results_r100p:
        print("No valid results found.")
        return

    mean_r100p = np.mean(results_r100p)
    std_r100p = np.std(results_r100p)
    mean_auc = np.mean(results_auc)
    std_auc = np.std(results_auc)
    
    print(f"Total chunks processed: {len(results_r100p)}")
    print(f"Sliding Window Protocol Mean R@100P: {mean_r100p:.2f}% ± {std_r100p:.2f}%")
    print(f"Sliding Window Protocol Mean AUC-PR: {mean_auc:.2f}% ± {std_auc:.2f}%")

if __name__ == "__main__":
    main()
