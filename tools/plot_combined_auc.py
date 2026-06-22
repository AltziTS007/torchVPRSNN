import sys
import os
ROOT_DIR_HACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

import os
import re
import matplotlib.pyplot as plt

def extract_auc_data(directory):
    chunk_sizes = []
    auc_means = []
    auc_stds = []
    
    if not os.path.exists(directory):
        return [], [], []

    for filename in os.listdir(directory):
        if filename.startswith("summary_") and filename.endswith(".txt"):
            cs = int(filename.replace("summary_", "").replace(".txt", ""))
            
            with open(os.path.join(directory, filename), 'r') as f:
                content = f.read()
                
            match_auc = re.search(r'Mean AUC-PR:\s*([\d\.]+)%\s*±\s*([\d\.]+)%', content)
            
            if match_auc:
                chunk_sizes.append(cs)
                auc_means.append(float(match_auc.group(1)))
                auc_stds.append(float(match_auc.group(2)))
                
    # Sort by chunk size
    if chunk_sizes:
        sorted_idx = sorted(range(len(chunk_sizes)), key=lambda k: chunk_sizes[k])
        chunk_sizes = [chunk_sizes[i] for i in sorted_idx]
        auc_means = [auc_means[i] for i in sorted_idx]
        auc_stds = [auc_stds[i] for i in sorted_idx]
        
    return chunk_sizes, auc_means, auc_stds

def main():
    nordland_dir = os.path.join(ROOT_DIR_HACK, "sliding_window_full_results")
    oxford_dir = os.path.join(ROOT_DIR_HACK, "sliding_window_full_results_oxford")
    
    n_cs, n_means, n_stds = extract_auc_data(nordland_dir)
    o_cs, o_means, o_stds = extract_auc_data(oxford_dir)
    
    if not n_cs and not o_cs:
        print("No results to plot.")
        return
        
    plt.figure(figsize=(8, 6))
    
    if n_cs:
        plt.errorbar(n_cs, n_means, yerr=n_stds, fmt='-o', color='teal', capsize=5, capthick=2, linewidth=2, markersize=8, label="Nordland AUC-PR")
        
    if o_cs:
        plt.errorbar(o_cs, o_means, yerr=o_stds, fmt='-s', color='darkorange', capsize=5, capthick=2, linewidth=2, markersize=8, label="Oxford AUC-PR")
    
    plt.title('Nordland & Oxford Datasets', fontsize=14)
    plt.xlabel('Number of Places', fontsize=12)
    plt.ylabel('AUC PR (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 105)
    plt.legend(loc="lower left", fontsize=12)
    
    # Save the plot
    save_path = 'ablation_sliding_window_combined_auc.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
