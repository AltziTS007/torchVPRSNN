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
        
    # Extract FLOPs
    flops_match = re.search(r'FLOPs/query:\s*([\d\.]+)', content)
    if flops_match:
        metrics['flops'] = float(flops_match.group(1))
        
    # Extract Effective Energy
    energy_match = re.search(r'Effective Energy per inference:\s*([\d\.]+)\s*J', content)
    if energy_match:
        metrics['effective_energy_j'] = float(energy_match.group(1))
        
    # Extract Latency
    latency_match = re.search(r'Inference time/query:\s*([\d\.]+)\s*ms', content)
    if latency_match:
        metrics['latency_ms'] = float(latency_match.group(1))
        
    # Extract R@100P for Weighted+Prob
    wp_matches = re.findall(r'Weighted\+Prob\s*\|.*', content)
    if wp_matches:
        cols = [col.strip().strip('%').strip() for col in wp_matches[-1].split('|')[1:]]
        if len(cols) >= 4:
            metrics['r100p_weighted_prob'] = float(cols[2])
        elif len(cols) >= 2:
            metrics['r100p_weighted_prob'] = float(cols[1])
        
    return metrics

def main():
    directory = "tsteps_ablation_results_oxford"
    results = {}
    
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

    if not results:
        print("No results found to plot.")
        return

    # Sort by t_steps
    t_steps_list = sorted(results.keys())
    latencies = [results[ts]['latency_ms'] for ts in t_steps_list]
    r100ps = [results[ts]['r100p_weighted_prob'] for ts in t_steps_list]
    energies = [results[ts]['effective_energy_j'] for ts in t_steps_list]

    # Plot 1: Latency vs R@100P
    plt.figure(figsize=(8, 6))
    plt.plot(latencies, r100ps, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    texts = []
    for i, txt in enumerate(t_steps_list):
        texts.append(plt.text(latencies[i], r100ps[i], f"t={txt}", ha='center', va='center'))
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
        
    plt.title('Latency vs. R@100P Ablation (Oxford RobotCar)', fontsize=14)
    plt.xlabel('Inference Latency per Query (ms)', fontsize=12)
    plt.ylabel('R@100P (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('ablation_latency_vs_r100p_oxford.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved plot to ablation_latency_vs_r100p_oxford.png")

    # Plot 2: Energy vs R@100P (if available)
    if all(e is not None for e in energies):
        plt.figure(figsize=(8, 6))
        plt.plot(energies, r100ps, marker='s', linestyle='-', color='g', linewidth=2, markersize=8)
        
        texts = []
        for i, txt in enumerate(t_steps_list):
            texts.append(plt.text(energies[i], r100ps[i], f"t={txt}", ha='center', va='center'))
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
            
        plt.title('Energy vs. R@100P Ablation (Oxford RobotCar)', fontsize=14)
        plt.xlabel('Effective Energy per Inference (Joules)', fontsize=12)
        plt.ylabel('R@100P (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('ablation_energy_vs_r100p_oxford.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved plot to ablation_energy_vs_r100p_oxford.png")

if __name__ == "__main__":
    main()
