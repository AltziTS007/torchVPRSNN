import torch
import numpy as np
import argparse

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

def run_table_v_for_seed(seed, max_samples=100, device="cuda", data_dir="nordland_clean"):
    set_seed(seed)
    
    import os
    train_path = [os.path.join(data_dir, "data/spring"), os.path.join(data_dir, "data/fall")]
    test_path = os.path.join(data_dir, "data/summer")
    
    train_loader, test_loader, num_classes = get_nordland_loaders(
        train_path=train_path, test_path=test_path,
        batch_size=64, max_samples=max_samples,
        start_idx=0, shuffle_train=True
    )
    
    t_steps = 150
    encoder = RateEncoder(t_steps=t_steps, rate_scale=0.25).to(device)
    net = torchVPRSNN(
        n_in=784, n_exc=400, t_steps=t_steps,
        beta_e=0.95, beta_i=0.90, thr_e_init=1.5, thr_i=1.0,
        a_plus=5e-4, a_minus=5e-6, tau_pre=15.0, tau_post=15.0,
        w_max=1.0, w_ei=1.0, w_ie=10.0, thr_eta=0.001, target_rate=0.01,
        max_samples=max_samples, device=device
    ).to(device)
    
    net.train()
    for epoch in range(1, 121):
        for xb, _ in train_loader:
            xb = xb.to(device)
            spk_in = encoder(xb)
            net(spk_in, do_stdp=True)
            
    assignments, _ = get_standard_assignments(net, encoder, train_loader, 400, num_classes, device)
    
    _, _, targets, _, _, S_Q_cpu, _ = evaluate_vpr(
        net, encoder, test_loader, assignments, num_classes, device, simulate_spillover=False
    )
    S_Q = S_Q_cpu.to(device)
    
    S_R = get_training_spike_counts(net, encoder, train_loader, 400, num_classes, device=device)
    scores_weighted = weighted_assignment_inference(S_Q, S_R, gamma=0.02)
    prob_scores = probability_based_assignment(scores_weighted)
    targets = np.array(targets)
    
    k_values = [1, 3, 5, 7, 10, 15]
    results = {}
    
    for k in k_values:
        agg, vt, _ = sliding_window_aggregation(prob_scores, targets, k=k, rule='product', velocity_factor=1.0)
        results[k] = calculate_r_at_100p(agg.cpu().numpy(), np.array(vt)) if len(vt) > 0 else 0.0
        print(f"Seed {seed}, k={k} -> R@100P: {results[k]:.2f}%")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Table V ablation (Nordland)")
    parser.add_argument("--data-dir", type=str, default="nordland_clean", help="Path to Nordland dataset root")
    args = parser.parse_args()

    seeds = range(16, 31) # 15 networks as specified in Table V
    k_values = [1, 3, 5, 7, 10, 15]
    all_res = {k: [] for k in k_values}
    
    for s in seeds:
        print(f"--- Running seed {s} ---")
        res = run_table_v_for_seed(s, data_dir=args.data_dir)
        for k in k_values:
            all_res[k].append(res[k])
            
    print("\n\n================================================================================")
    print("TABLE V: SEQUENTIAL AGGREGATION: MEAN R@100P VS. WINDOW SIZE k")
    print("(NORDLAND, 100 PLACES, CONSTANT-VELOCITY TRAVERSAL, PRODUCT RULE)")
    print("================================================================================")
    print(f"{'k':<5} | {'Mean R@100P (%)':<15} | {'Std. Dev. (%)':<15}")
    print("-" * 45)
    
    for k in k_values:
        mean_val = np.mean(all_res[k])
        std_val = np.std(all_res[k])
        print(f"{k:<5} | {mean_val:>6.2f}          | {std_val:>5.2f}")
