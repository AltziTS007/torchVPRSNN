import os
import re
import matplotlib.pyplot as plt
import numpy as np

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

def plot_gradient_dataset(directory, title_suffix, out_prefix):
    results = {}
    if not os.path.exists(directory):
        return
        
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
        print(f"No results found in {directory}")
        return

    t_steps_list = sorted(results.keys())
    latencies = [results[ts]['latency_ms'] for ts in t_steps_list]
    r100ps = [results[ts]['r100p_weighted_prob'] for ts in t_steps_list]
    energies = [results[ts]['effective_energy_j'] for ts in t_steps_list]

    cmap = 'viridis'

    # Plot Latency
    plt.figure(figsize=(9, 6))
    plt.plot(latencies, r100ps, color='gray', alpha=0.5, linestyle='-', linewidth=2, zorder=1)
    scatter = plt.scatter(latencies, r100ps, c=t_steps_list, cmap=cmap, s=150, zorder=2, edgecolors='black', linewidth=0.8)
    
    # Add start/end annotations
    plt.annotate("Start\n(t=20)", (latencies[0], r100ps[0]), textcoords="offset points", xytext=(-30, -20), ha='center', fontsize=10, arrowprops=dict(arrowstyle="->", color='gray'))
    plt.annotate("End\n(t=1000)", (latencies[-1], r100ps[-1]), textcoords="offset points", xytext=(30, 20), ha='center', fontsize=10, arrowprops=dict(arrowstyle="->", color='gray'))

    cbar = plt.colorbar(scatter)
    cbar.set_label('Simulation Time $t$ (steps)', fontsize=12)
    plt.title(f'Latency vs. R@100P Ablation {title_suffix}', fontsize=15, pad=15)
    plt.xlabel('Inference Latency per Query (ms)', fontsize=13)
    plt.ylabel('R@100P (%)', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_latency.png', dpi=300)
    plt.close()

    # Plot Energy
    if all(e is not None for e in energies):
        plt.figure(figsize=(9, 6))
        plt.plot(energies, r100ps, color='gray', alpha=0.5, linestyle='-', linewidth=2, zorder=1)
        scatter = plt.scatter(energies, r100ps, c=t_steps_list, cmap=cmap, s=150, zorder=2, edgecolors='black', linewidth=0.8)
        
        plt.annotate("Start\n(t=20)", (energies[0], r100ps[0]), textcoords="offset points", xytext=(-30, -20), ha='center', fontsize=10, arrowprops=dict(arrowstyle="->", color='gray'))
        plt.annotate("End\n(t=1000)", (energies[-1], r100ps[-1]), textcoords="offset points", xytext=(30, 20), ha='center', fontsize=10, arrowprops=dict(arrowstyle="->", color='gray'))

        cbar = plt.colorbar(scatter)
        cbar.set_label('Simulation Time $t$ (steps)', fontsize=12)
        plt.title(f'Energy vs. R@100P Ablation {title_suffix}', fontsize=15, pad=15)
        plt.xlabel('Effective Energy per Inference (Joules)', fontsize=13)
        plt.ylabel('R@100P (%)', fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'{out_prefix}_energy.png', dpi=300)
        plt.close()
        
    print(f"Generated gradient plots for {title_suffix}")

def main():
    plot_gradient_dataset("tsteps_ablation_results", "(Nordland)", "ablation_gradient_nordland")
    plot_gradient_dataset("tsteps_ablation_results_oxford", "(Oxford RobotCar)", "ablation_gradient_oxford")

if __name__ == "__main__":
    main()
