"""
snn_model.py

Unsupervised Spiking Neural Network for Visual Place Recognition (VPR)
based on a Diehl & Cook-style architecture with STDP learning.

Key characteristics:
- Rate-based Poisson input encoding
- Excitatory / inhibitory populations
- Hard Winner-Take-All (WTA)
- Weight-dependent STDP (no backprop)
- Homeostatic threshold adaptation
- Evaluation via neuron-to-place assignment

Designed for:
- Nordland dataset (seasonal VPR)
- Research-grade experimentation, not production

REFACTORED:
Now uses modular imports from:
- encoders.py
- mechanisms.py
- networks.py
- utils.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
import numpy as np
import datetime
import json

# ============================================================
# NEW IMPORTS
# ============================================================
from encoders import RateEncoder
from networks import torchVPRSNN
from utils import Logger, plot_stdp_monitor

# VPR-specific evaluation utilities
from vprsnn_evaluation import (
    get_nordland_loaders, 
    get_standard_assignments, 
    evaluate_vpr, 
    plot_weights,
    calculate_p_at_100r,
    calculate_r_at_100p,
    plot_distance_matrix,
    plot_pr_curve,
    plot_recall_at_n,
    plot_neuron_assignments,
    plot_neuron_assignments,
    visualize_qualitative_results
)

from neuronal_assignments import (
    get_training_spike_counts,
    weighted_assignment_inference,
    probability_based_assignment
)


# ============================================================
# MAIN EXPERIMENT PIPELINE
# ============================================================

def main():
    """
    End-to-end experiment:
        1. Load Nordland dataset
        2. Unsupervised STDP training
        3. Neuron-to-place assignment
        4. Cross-season VPR evaluation
        5. Metric visualization
    """

    # --------------------------------------------------------
    # Hyperparameters & Configuration
    # --------------------------------------------------------
    params = {
        # Data paths
        "TRAIN_PATH": ["nordland_clean/data/spring", "nordland_clean/data/fall"],
        "TEST_PATH": "nordland_clean/data/summer",
        
        # Training loop
        "EPOCHS": 120,
        "BATCH_SIZE": 64,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        
        # Architecture
        "N_IN": 784,
        "N_EXC": 200,
        
        # Rate Encoder
        "t_steps": 100,
        "rate_scale": 0.25,
        
        # Neuron / STDP Params (Diehl & Cook)
        "beta_e": 0.95,
        "beta_i": 0.90,
        "thr_e_init": 1.5,
        "thr_i": 1.0,
        "a_plus": 5e-4,
        "a_minus": 5e-6,
        "tau_pre": 15.0,
        "tau_post": 15.0,
        "w_max": 1.0,
        "w_ei": 1.0,
        "w_ie": 10.0,
        "thr_eta": 0.001,
        "target_rate": 0.01
    }

    DEVICE = params["DEVICE"]
    print(f"Running on device: {DEVICE}")
    
    # --------------------------------------------------------
    # Results Directory Setup
    # --------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results/results_{timestamp}"
    weights_dir = os.path.join(results_dir, "weights_history")
    stdp_dir = os.path.join(results_dir, "stdp_viz")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(stdp_dir, exist_ok=True)
    
    print(f"Saving all results to: {results_dir}")

    # --------------------------------------------------------
    # Logger Setup (Redirect stdout)
    # --------------------------------------------------------
    sys.stdout = Logger(os.path.join(results_dir, "output.log"))
    print(f"Logging execution output to: {os.path.join(results_dir, 'output.log')}")

    # Save hyperparameters
    with open(os.path.join(results_dir, "hyperparameters.json"), "w") as f:
        json.dump(params, f, indent=4)
    print("Saved hyperparameters.json")

    # --------------------------------------------------------
    # Data loading
    # --------------------------------------------------------
    try:
        train_loader, test_loader, num_classes = get_nordland_loaders(
            train_path=params["TRAIN_PATH"],
            test_path=params["TEST_PATH"],
            batch_size=params["BATCH_SIZE"],
            max_samples=100,
            shuffle_train=True
        )
    except Exception as e:
        print("Dataset loading failed:", e)
        return

    # --------------------------------------------------------
    # Model initialization
    # --------------------------------------------------------
    encoder = RateEncoder(
        t_steps=params["t_steps"], 
        rate_scale=params["rate_scale"]
    ).to(DEVICE)
    
    # Using the new torchVPRSNN class
    net = torchVPRSNN(
        n_in=params["N_IN"],
        n_exc=params["N_EXC"],
        t_steps=params["t_steps"],
        beta_e=params["beta_e"],
        beta_i=params["beta_i"],
        thr_e_init=params["thr_e_init"],
        thr_i=params["thr_i"],
        a_plus=params["a_plus"],
        a_minus=params["a_minus"],
        tau_pre=params["tau_pre"],
        tau_post=params["tau_post"],
        w_max=params["w_max"],
        w_ei=params["w_ei"],
        w_ie=params["w_ie"],
        thr_eta=params["thr_eta"],
        target_rate=params["target_rate"],
        device=DEVICE
    ).to(DEVICE)

    # --------------------------------------------------------
    # Unsupervised STDP training
    # --------------------------------------------------------
    print("Starting STDP training...")
    net.train()

    total_batches = len(train_loader)

    for epoch in range(1, params["EPOCHS"] + 1):
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{params['EPOCHS']}")
            
        for batch_idx, (xb, _) in enumerate(train_loader):
            xb = xb.to(DEVICE)
            spk_in = encoder(xb)
            
            # Monitor only on last batch
            is_last_batch = (batch_idx == total_batches - 1)
            
            if is_last_batch:
                _, history = net(spk_in, do_stdp=True, monitor=True)
                plot_stdp_monitor(history, epoch, stdp_dir)
            else:
                _ = net(spk_in, do_stdp=True)
        
        # Save weights every 10 epochs
        if epoch % 10 == 0:
            epoch_weight_path = os.path.join(weights_dir, f"epoch_{epoch}_weights.png")
            plot_weights(net.w_in_exc, params["N_EXC"], save_path=epoch_weight_path)

    # --------------------------------------------------------
    # Neuron-place assignment
    # --------------------------------------------------------
    assignments, avg_rates = get_standard_assignments(
        net=net,
        encoder=encoder,
        loader=train_loader,
        n_exc=params["N_EXC"],
        n_classes=num_classes,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    acc, preds, targets, sim_matrix, prob_matrix, S_Q_cpu = evaluate_vpr(
        net=net,
        encoder=encoder,
        loader=test_loader,
        assignments=assignments,
        n_classes=num_classes,
        device=DEVICE
    )
    S_Q = S_Q_cpu.to(DEVICE)

    # --------------------------------------------------------
    # NEW: Weighted & Probability-based Assignments
    # --------------------------------------------------------
    print("\n--- Computing Weighted Neuronal Assignments ---")
    
    # 1. Get Training Spike Counts S^R
    S_R = get_training_spike_counts(net, encoder, train_loader, params["N_EXC"], num_classes, device=DEVICE)
    
    # 2. Get Query Spike Counts S^Q
    # S_Q provided by evaluate_vpr to ensure alignment with targets
    print(f"Using Query Spike Counts (S^Q) from evaluation step. Shape: {S_Q.shape}")
            
    # 3. Weighted Assignment
    scores_weighted = weighted_assignment_inference(S_Q, S_R, gamma=0.02)
    
    # 4. Probability-based (on Weighted)
    scores_prob = probability_based_assignment(scores_weighted)
    
    # Accuracies for table
    def calc_acc(scores, targets):
        preds = scores.argmax(dim=1).cpu().numpy()
        correct = (preds == targets).sum()
        return 100 * correct / len(targets)
        
    acc_w = calc_acc(scores_weighted, targets)
    acc_p = calc_acc(scores_prob, targets)

    # 6. Unified Result Processing & Plotting
    # --------------------------------------------------------
    # Handle ConcatDataset for training (use first dataset as reference for viz)
    vis_train_ds = train_loader.dataset
    if isinstance(vis_train_ds, torch.utils.data.ConcatDataset):
        vis_train_ds = vis_train_ds.datasets[0]

    methods = [
        ("Standard", sim_matrix, preds, acc),         # Standard
        ("Weighted", scores_weighted.cpu().numpy(), scores_weighted.argmax(dim=1).cpu().numpy(), acc_w), # Weighted
        ("Weighted+Prob", scores_prob.cpu().numpy(), scores_prob.argmax(dim=1).cpu().numpy(), acc_p)     # Weighted+Prob
    ]
    
    # Store metrics for final table
    final_metrics = []

    for name, scores, pred_labels, accuracy in methods:
        print(f"\nProcessing Results for: {name}...")
        
        # Create Method Directory
        method_dir = os.path.join(results_dir, name.replace(" ", "_"))
        os.makedirs(method_dir, exist_ok=True)
        
        # Calculate Metrics
        p100r = calculate_p_at_100r(scores, targets)
        r100p = calculate_r_at_100p(scores, targets)
        
        final_metrics.append({
            "Method": name,
            "Accuracy": accuracy,
            "P@100R": p100r,
            "R@100P": r100p
        })
        
        # Visualizations
        # 1. Distance Matrix
        plot_distance_matrix(scores, save_path=os.path.join(method_dir, "distance_matrix.png"), targets=targets)
        
        # 2. PR Curve
        plot_pr_curve(scores, targets, num_classes, save_path=os.path.join(method_dir, "pr_curve.png"))
        
        # 3. Recall @ N
        plot_recall_at_n(scores, targets, save_path=os.path.join(method_dir, "recall_at_n.png"))
        
        # 4. Qualitative Results (Reuse Standard Assignments for neuron interpretation, but use Method Preds)
        # Note: Assignments are static (Standard Training), but Preds change per method.
        visualize_qualitative_results(
            query_ds=test_loader.dataset,
            database_ds=vis_train_ds,
            assignments=assignments,
            weights=net.w_in_exc,
            preds=pred_labels,
            targets=targets,
            save_path=os.path.join(method_dir, "qualitative.png")
        )
        # Also plot assignments logic just once or copy? 
        # Since assignments are common (Standard Training), we saved them in root.
    
    # --------------------------------------------------------
    # 7. Final Summary Table
    # --------------------------------------------------------
    print("\n" + "="*80)
    
    # Check if Accuracy == P@100R for ALL methods (or ANY? User: "if accuracy is the same as p@100r remove it")
    # I'll check if they are close for all methods. if so, drop Acc.
    redundant_acc = all(abs(m["Accuracy"] - m["P@100R"]) < 0.1 for m in final_metrics)
    
    if redundant_acc:
        header = f"{'METHOD':<30} | {'P@100R':<12} | {'R@100P':<10}"
        print(f" (Note: Accuracy removed as it is identical to P@100R)")
    else:
        header = f"{'METHOD':<30} | {'ACCURACY':<10} | {'P@100R':<12} | {'R@100P':<10}"
        
    print(header)
    print("-" * 80)
    
    for m in final_metrics:
        if redundant_acc:
             print(f"{m['Method']:<30} | {m['P@100R']:<11.2f}% | {m['R@100P']:<9.2f}%")
        else:
             print(f"{m['Method']:<30} | {m['Accuracy']:<9.2f}% | {m['P@100R']:<11.2f}% | {m['R@100P']:<9.2f}%")
             
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
