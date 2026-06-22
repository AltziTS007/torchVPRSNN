import os
import re
import matplotlib.pyplot as plt
from adjustText import adjust_text

def parse_log(filepath):
    metrics = {
        'flops': None,
        'effective_energy_j': None,
        'latency_ms': None,
        'r100p_weighted_prob': None
    }
    
    with open(filepath, 'r') as f:
        content = f.read()
        
    flops_match = re.search(r'FLOPs/query:\s*([\d\.]+)', content)
    if flops_match:
        metrics['flops'] = float(flops_match.group(1))
        
    energy_match = re.search(r'Effective Energy per inference:\s*([\d\.]+)\s*J', content)
    if energy_match:
        metrics['effective_energy_j'] = float(energy_match.group(1))
        
    latency_match = re.search(r'Inference time/query:\s*([\d\.]+)\s*ms', content)
    if latency_match:
        metrics['latency_ms'] = float(latency_match.group(1))
        
    wp_matches = re.findall(r'Weighted\+Prob\s*\|.*', content)
    if wp_matches:
        cols = [col.strip().strip('%').strip() for col in wp_matches[-1].split('|')[1:]]
        if len(cols) >= 4:
            metrics['r100p_weighted_prob'] = float(cols[2])
        elif len(cols) >= 2:
            metrics['r100p_weighted_prob'] = float(cols[1])
            
    return metrics

def get_results(directory):
    results = {}
    if not os.path.exists(directory):
        return results
        
    for filename in os.listdir(directory):
        if filename.startswith("tsteps_") and filename.endswith(".log"):
            t_steps_str = filename.replace("tsteps_", "").replace(".log", "")
            try:
                t_steps = int(t_steps_str)
                metrics = parse_log(os.path.join(directory, filename))
                if metrics['r100p_weighted_prob'] is not None:
                    results[t_steps] = metrics
            except ValueError:
                continue
    return results

def main():
    res_nordland = get_results("tsteps_ablation_results")
    res_oxford = get_results("tsteps_ablation_results_oxford")

    # Filter to common t_steps if needed, or just plot what exists. 
    # Let's plot what exists for each.
    t_nordland = sorted(res_nordland.keys())
    lat_nordland = [res_nordland[t]['latency_ms'] for t in t_nordland]
    r100p_nordland = [res_nordland[t]['r100p_weighted_prob'] for t in t_nordland]
    en_nordland = [res_nordland[t]['effective_energy_j'] for t in t_nordland]

    t_oxford = sorted(res_oxford.keys())
    lat_oxford = [res_oxford[t]['latency_ms'] for t in t_oxford]
    r100p_oxford = [res_oxford[t]['r100p_weighted_prob'] for t in t_oxford]
    en_oxford = [res_oxford[t]['effective_energy_j'] for t in t_oxford]

    # --- Plot Latency ---
    plt.figure(figsize=(9, 6))
    
    plt.plot(lat_nordland, r100p_nordland, marker='o', linestyle='-', color='tab:blue', linewidth=2, markersize=8, label='Nordland (Seasonal)')
    plt.plot(lat_oxford, r100p_oxford, marker='s', linestyle='-', color='tab:red', linewidth=2, markersize=8, label='Oxford (Urban Dynamic)')
    
    texts = []
    for i, t in enumerate(t_nordland):
        texts.append(plt.text(lat_nordland[i], r100p_nordland[i], f"t={t}", ha='center', va='center', color='tab:blue', fontsize=9))
    for i, t in enumerate(t_oxford):
        texts.append(plt.text(lat_oxford[i], r100p_oxford[i], f"t={t}", ha='center', va='center', color='tab:red', fontsize=9))
        
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
    
    plt.title('Latency vs. R@100P Ablation (Combined)', fontsize=15, pad=15)
    plt.xlabel('Inference Latency per Query (ms)', fontsize=13)
    plt.ylabel('R@100P (%)', fontsize=13)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('ablation_combined_latency.png', dpi=300)
    plt.close()

    # --- Plot Energy ---
    plt.figure(figsize=(9, 6))
    
    plt.plot(en_nordland, r100p_nordland, marker='o', linestyle='-', color='tab:blue', linewidth=2, markersize=8, label='Nordland (Seasonal)')
    plt.plot(en_oxford, r100p_oxford, marker='s', linestyle='-', color='tab:red', linewidth=2, markersize=8, label='Oxford (Urban Dynamic)')
    
    texts = []
    for i, t in enumerate(t_nordland):
        texts.append(plt.text(en_nordland[i], r100p_nordland[i], f"t={t}", ha='center', va='center', color='tab:blue', fontsize=9))
    for i, t in enumerate(t_oxford):
        texts.append(plt.text(en_oxford[i], r100p_oxford[i], f"t={t}", ha='center', va='center', color='tab:red', fontsize=9))
        
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
    
    plt.title('Energy vs. R@100P Ablation (Combined)', fontsize=15, pad=15)
    plt.xlabel('Effective Energy per Inference (Joules)', fontsize=13)
    plt.ylabel('R@100P (%)', fontsize=13)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('ablation_combined_energy.png', dpi=300)
    plt.close()

    print("Generated combined plots.")

if __name__ == "__main__":
    main()
