import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import sys

# Import SNN model components
from snn_model import set_seed
from encoders import RateEncoder
from networks import torchVPRSNN
from vprsnn_evaluation import get_nordland_loaders, evaluate_vpr, calculate_r_at_100p, get_standard_assignments
from neuronal_assignments import (
    get_training_spike_counts,
    weighted_assignment_inference,
    probability_based_assignment,
    sliding_window_aggregation
)

def run_ablation_for_seed(seed, max_samples=100, device="cuda", data_dir="nordland_clean"):
    set_seed(seed)
    
    # Setup DataLoaders
    import os
    train_path = [os.path.join(data_dir, "data/spring"), os.path.join(data_dir, "data/fall")]
    test_path = os.path.join(data_dir, "data/summer")
    
    train_loader, test_loader, num_classes = get_nordland_loaders(
        train_path=train_path,
        test_path=test_path,
        batch_size=64,
        max_samples=max_samples,
        start_idx=0,
        shuffle_train=True
    )
    
    # Initialize Model
    t_steps = 150
    encoder = RateEncoder(t_steps=t_steps, rate_scale=0.25).to(device)
    
    net = torchVPRSNN(
        n_in=784, n_exc=400, t_steps=t_steps,
        beta_e=0.95, beta_i=0.90, thr_e_init=1.5, thr_i=1.0,
        a_plus=5e-4, a_minus=5e-6, tau_pre=15.0, tau_post=15.0,
        w_max=1.0, w_ei=1.0, w_ie=10.0, thr_eta=0.001, target_rate=0.01,
        max_samples=max_samples, device=device
    ).to(device)
    
    # Train STDP
    net.train()
    for epoch in range(1, 121):
        for xb, _ in train_loader:
            xb = xb.to(device)
            spk_in = encoder(xb)
            net(spk_in, do_stdp=True)
            
    # Neuron Assignments
    assignments, _ = get_standard_assignments(net, encoder, train_loader, 400, num_classes, device)
    
    # Evaluate Baseline
    _, _, targets, _, _, S_Q_cpu, _ = evaluate_vpr(
        net, encoder, test_loader, assignments, num_classes, device, simulate_spillover=False
    )
    S_Q = S_Q_cpu.to(device)
    
    # Weighted Assignments -> Probabilities
    S_R = get_training_spike_counts(net, encoder, train_loader, 400, num_classes, device=device)
    scores_weighted = weighted_assignment_inference(S_Q, S_R, gamma=0.02)
    prob_scores = probability_based_assignment(scores_weighted)
    
    targets = np.array(targets)
    
    results = {}
    
    # 1. Baseline
    agg, vt, _ = sliding_window_aggregation(prob_scores, targets, k=5, rule='product', velocity_factor=1.0)
    results['baseline'] = calculate_r_at_100p(agg.cpu().numpy(), np.array(vt)) if len(vt) > 0 else 0.0
    
    # 2. Velocity Mismatch (+-10%)
    agg_11, vt_11, _ = sliding_window_aggregation(prob_scores, targets, k=5, rule='product', velocity_factor=1.1)
    results['vel_1_10'] = calculate_r_at_100p(agg_11.cpu().numpy(), np.array(vt_11)) if len(vt_11) > 0 else 0.0

    agg_09, vt_09, _ = sliding_window_aggregation(prob_scores, targets, k=5, rule='product', velocity_factor=0.9)
    results['vel_0_90'] = calculate_r_at_100p(agg_09.cpu().numpy(), np.array(vt_09)) if len(vt_09) > 0 else 0.0
    
    # 3. Velocity Mismatch (+-20%)
    agg_12, vt_12, _ = sliding_window_aggregation(prob_scores, targets, k=5, rule='product', velocity_factor=1.2)
    results['vel_1_20'] = calculate_r_at_100p(agg_12.cpu().numpy(), np.array(vt_12)) if len(vt_12) > 0 else 0.0

    agg_08, vt_08, _ = sliding_window_aggregation(prob_scores, targets, k=5, rule='product', velocity_factor=0.8)
    results['vel_0_80'] = calculate_r_at_100p(agg_08.cpu().numpy(), np.array(vt_08)) if len(vt_08) > 0 else 0.0

    # 4. Velocity Mismatch (+-50%)
    agg_15, vt_15, _ = sliding_window_aggregation(prob_scores, targets, k=5, rule='product', velocity_factor=1.5)
    results['vel_1_50'] = calculate_r_at_100p(agg_15.cpu().numpy(), np.array(vt_15)) if len(vt_15) > 0 else 0.0

    agg_05, vt_05, _ = sliding_window_aggregation(prob_scores, targets, k=5, rule='product', velocity_factor=0.5)
    results['vel_0_50'] = calculate_r_at_100p(agg_05.cpu().numpy(), np.array(vt_05)) if len(vt_05) > 0 else 0.0
    
    # 5. Reverse Traversal
    targets_rev = list(reversed(targets))
    prob_scores_rev = torch.flip(prob_scores, dims=[0])
    agg_rev, vt_rev, _ = sliding_window_aggregation(prob_scores_rev, targets_rev, k=5, rule='product', velocity_factor=1.0)
    results['reverse'] = calculate_r_at_100p(agg_rev.cpu().numpy(), np.array(vt_rev)) if len(vt_rev) > 0 else 0.0
    
    # 6. Dropped Frames (10%)
    np.random.seed(seed) # reuse seed for deterministic drops
    keep_10 = [i for i in range(len(targets)) if np.random.rand() > 0.10]
    prob_scores_drop_10 = prob_scores[keep_10]
    targets_drop_10 = [targets[i] for i in keep_10]
    agg_d10, vt_d10, _ = sliding_window_aggregation(prob_scores_drop_10, targets_drop_10, k=5, rule='product', velocity_factor=1.0)
    results['drop_10'] = calculate_r_at_100p(agg_d10.cpu().numpy(), np.array(vt_d10)) if len(vt_d10) > 0 else 0.0
    
    # 7. Dropped Frames (50%)
    np.random.seed(seed + 1)
    keep_50 = [i for i in range(len(targets)) if np.random.rand() > 0.50]
    prob_scores_drop_50 = prob_scores[keep_50]
    targets_drop_50 = [targets[i] for i in keep_50]
    agg_d50, vt_d50, _ = sliding_window_aggregation(prob_scores_drop_50, targets_drop_50, k=5, rule='product', velocity_factor=1.0)
    results['drop_50'] = calculate_r_at_100p(agg_d50.cpu().numpy(), np.array(vt_d50)) if len(vt_d50) > 0 else 0.0
    
    # 8. Route Deviations
    targets_dev = list(targets)
    prob_scores_dev = prob_scores.clone()
    for i in range(15, len(targets), 15):
        jump_idx = i + 5
        if jump_idx < len(targets):
            targets_dev[i] = targets[jump_idx]
            prob_scores_dev[i] = prob_scores[jump_idx]
    agg_dev, vt_dev, _ = sliding_window_aggregation(prob_scores_dev, targets_dev, k=5, rule='product', velocity_factor=1.0)
    results['route_dev'] = calculate_r_at_100p(agg_dev.cpu().numpy(), np.array(vt_dev)) if len(vt_dev) > 0 else 0.0
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Table IV ablation (Nordland)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nordland_clean"), help="Path to Nordland dataset root")
    args = parser.parse_args()

    seeds = range(16, 26) # 10 seeds as in previous scripts
    all_res = {k: [] for k in ['baseline', 'vel_1_10', 'vel_0_90', 'vel_1_20', 'vel_0_80', 'vel_1_50', 'vel_0_50', 'reverse', 'drop_10', 'drop_50', 'route_dev']}
    
    for s in seeds:
        print(f"Running seed {s}...")
        res = run_ablation_for_seed(s, data_dir=args.data_dir)
        for k in res:
            all_res[k].append(res[k])
            
    output_lines = []
    output_lines.append("\n\n" + "="*80)
    output_lines.append("TABLE IV: OPERATIONAL ENVELOPE OF SEQUENCE AGGREGATION (k = 5, PRODUCT RULE)")
    output_lines.append("NORDLAND, 100 PLACES (Means over 10 seeds: 16-25)")
    output_lines.append("="*80)
    output_lines.append(f"{'Condition':<30} | {'Mean R@100P (%)':<15} | {'Std Dev':<10}")
    output_lines.append("-" * 60)
    
    # Map to table names
    mapping = [
        ("Baseline (1.0x)", "baseline"),
        ("Velocity mismatch (+10%)", "vel_1_10"),
        ("Velocity mismatch (-10%)", "vel_0_90"),
        ("Velocity mismatch (+20%)", "vel_1_20"),
        ("Velocity mismatch (-20%)", "vel_0_80"),
        ("Velocity mismatch (+50%)", "vel_1_50"),
        ("Velocity mismatch (-50%)", "vel_0_50"),
        ("Reverse traversal", "reverse"),
        ("10% dropped frames", "drop_10"),
        ("50% dropped frames", "drop_50"),
        ("Route deviations", "route_dev"),
    ]
    
    for name, key in mapping:
        mean_val = np.mean(all_res[key])
        std_val = np.std(all_res[key])
        output_lines.append(f"{name:<30} | {mean_val:>6.2f}          | ±{std_val:>5.2f}")
    
    # Combined means for +- mismatches to match table exactly
    output_lines.append("\n--- Combined Velocity Mismatches ---")
    vel_10 = all_res['vel_1_10'] + all_res['vel_0_90']
    vel_20 = all_res['vel_1_20'] + all_res['vel_0_80']
    vel_50 = all_res['vel_1_50'] + all_res['vel_0_50']
    
    output_lines.append(f"{'Velocity mismatch (±10%)':<30} | {np.mean(vel_10):>6.2f}          | ±{np.std(vel_10):>5.2f}")
    output_lines.append(f"{'Velocity mismatch (±20%)':<30} | {np.mean(vel_20):>6.2f}          | ±{np.std(vel_20):>5.2f}")
    output_lines.append(f"{'Velocity mismatch (±50%)':<30} | {np.mean(vel_50):>6.2f}          | ±{np.std(vel_50):>5.2f}")

    final_output = "\n".join(output_lines)
    print(final_output)

    with open("table_iv_nordland_results.txt", "w") as f:
        f.write(final_output)
        
    print("Results saved to table_iv_nordland_results.txt")

