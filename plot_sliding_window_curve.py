import os
import re
import matplotlib.pyplot as plt

def main():
    directory = "sliding_window_full_results"
    
    chunk_sizes = []
    #r100p_means = []
    #r100p_stds = []
    auc_means = []
    auc_stds = []
    
    for filename in os.listdir(directory):
        if filename.startswith("summary_") and filename.endswith(".txt"):
            cs = int(filename.replace("summary_", "").replace(".txt", ""))
            
            with open(os.path.join(directory, filename), 'r') as f:
                content = f.read()
                
            #match_r100p = re.search(r'Mean R@100P:\s*([\d\.]+)%\s*±\s*([\d\.]+)%', content)
            match_auc = re.search(r'Mean AUC-PR:\s*([\d\.]+)%\s*±\s*([\d\.]+)%', content)
            
            if match_auc:
                chunk_sizes.append(cs)
                #r100p_means.append(float(match_r100p.group(1)))
                #r100p_stds.append(float(match_r100p.group(2)))
                auc_means.append(float(match_auc.group(1)))
                auc_stds.append(float(match_auc.group(2)))
                
    if not chunk_sizes:
        print("No results to plot.")
        return
        
    # Sort by chunk size
    sorted_idx = sorted(range(len(chunk_sizes)), key=lambda k: chunk_sizes[k])
    chunk_sizes = [chunk_sizes[i] for i in sorted_idx]
    #r100p_means = [r100p_means[i] for i in sorted_idx]
    #r100p_stds = [r100p_stds[i] for i in sorted_idx]
    auc_means = [auc_means[i] for i in sorted_idx]
    auc_stds = [auc_stds[i] for i in sorted_idx]
    
    plt.figure(figsize=(8, 6))
    
    # Plot R@100P
    #plt.errorbar(chunk_sizes, r100p_means, yerr=r100p_stds, fmt='-o', color='purple', capsize=5, capthick=2, linewidth=2, markersize=8, label="R@100P")
    
    # Plot AUC-PR
    plt.errorbar(chunk_sizes, auc_means, yerr=auc_stds, fmt='-s', color='teal', capsize=5, capthick=2, linewidth=2, markersize=8, label="AUC-PR")
    
    plt.title('Nordland', fontsize=14)
    plt.xlabel('Number of Places', fontsize=12)
    plt.ylabel('AUC PR (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 105)
    plt.legend(loc="lower left", fontsize=12)
    
    # Save the plot
    plt.savefig('ablation_sliding_window_curve.png', dpi=300, bbox_inches='tight')
    print("Saved plot to ablation_sliding_window_curve.png")

if __name__ == "__main__":
    main()
