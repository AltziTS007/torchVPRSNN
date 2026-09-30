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
import argparse
import random

def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from encoders import RateEncoder
from networks import torchVPRSNN
from utils import Logger, plot_stdp_monitor

from vprsnn_evaluation import (
    get_nordland_loaders, 
    get_mddrobots_loaders,
    get_mddrobots_room_loaders,
    get_standard_assignments, 
    evaluate_vpr, 
    plot_weights,
    calculate_p_at_100r,
    calculate_r_at_100p,
    calculate_auc_pr,
    plot_distance_matrix,
    plot_pr_curve,
    plot_recall_at_n,
    plot_neuron_assignments,
    visualize_qualitative_results
)

from neuronal_assignments import (
    get_training_spike_counts,
    weighted_assignment_inference,
    probability_based_assignment,
    sliding_window_aggregation
)


# ============================================================
# MAIN EXPERIMENT PIPELINE
# ============================================================

def _str_to_bool(value):
    """Parse common string forms into booleans for CLI flags."""
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_args():
    """CLI switches for runtime behavior."""
    parser = argparse.ArgumentParser(description="Train/evaluate VPR SNN")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Compute device: cpu, cuda, or auto (default).",
    )
    parser.add_argument(
        "--save-results",
        type=_str_to_bool,
        default=None,
        help="Enable/disable writing logs, json and images (true/false).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override MAX_SAMPLES for dataset truncation.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index for sliding window dataset evaluation.",
    )
    parser.add_argument(
        "--dataset",
        choices=["nordland", "oxford", "mddrobots", "mddrobots_rooms"],
        default="nordland",
        help="Dataset to use: 'nordland' (default), 'oxford', 'mddrobots', or 'mddrobots_rooms'.",
    )
    parser.add_argument(
        "--descriptor",
        choices=["raw", "dog", "spatial_pyramid"],
        default="raw",
        help="Input descriptor: 'raw' (default pixels+PatchNorm), 'dog' (Difference-of-Gaussians), 'spatial_pyramid' (spatial pyramid+HOG).",
    )
    parser.add_argument(
        "--room",
        type=str,
        default="Corridor1_RGB",
        help="MDDRobots room name (e.g., Corridor1_RGB, D3A_RGB, F102_RGB). Only used with --dataset mddrobots.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Take every Nth frame for subsampling (default: 4 for mddrobots, 1 otherwise).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=28,
        help="Image resolution. 28 -> 28x28 images, 56 -> 56x56 images.",
    )
    parser.add_argument(
        "--n-exc",
        type=int,
        default=400,
        help="Number of excitatory neurons (default: 400). Consider increasing for higher resolutions.",
    )
    parser.add_argument(
        "--thr-e-init",
        type=float,
        default=None,
        help="Initial excitatory threshold. Auto-scales with resolution if not provided.",
    )
    parser.add_argument(
        "--a-plus",
        type=float,
        default=None,
        help="STDP learning rate. Auto-scales with resolution if not provided.",
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        default=0.01,
        help="Homeostasis target firing rate (probability of spike per timestep).",
    )
    parser.add_argument(
        "--thr-eta",
        type=float,
        default=0.001,
        help="Homeostasis learning rate (how fast thresholds adapt).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=15,
        help="Maximum sequence length for the sliding window sweep.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="Tolerance window for ground truth matches (e.g., +/- N frames).",
    )
    parser.add_argument(
        "--wta-mode",
        choices=["hard", "soft", "none"],
        default="hard",
        help="WTA mode: hard (default), soft, or none.",
    )
    parser.add_argument(
        "--disable-homeostasis",
        action="store_true",
        help="Disable homeostatic threshold adaptation.",
    )
    parser.add_argument(
        "--disable-patch-norm",
        action="store_true",
        help="Disable weight patch normalization.",
    )
    parser.add_argument(
        "--standard-assignment-only",
        action="store_true",
        help="Only evaluate using standard assignments.",
    )
    parser.add_argument(
        "--simulate-spillover",
        action="store_true",
        help="Enable continuous ODE state carry-over (Ablation Study).",
    )
    parser.add_argument(
        "--sample-10m",
        action="store_true",
        help="Enable 10-meter spatial sampling for the Oxford dataset.",
    )
    parser.add_argument(
        "--t-steps",
        type=int,
        default=None,
        help="Override t_steps for the simulation duration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Path to save the trained model checkpoint (e.g. baseline.pt).",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to load a trained model checkpoint (skips STDP training).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom prefix for the results directory name.",
    )
    return parser.parse_args()

def calc_acc(scores, targets, tol=0):
    """Calculate accuracy with optional tolerance."""
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    preds = np.argmax(scores, axis=1)
    if tol == 0:
        correct = np.sum(preds == targets)
    else:
        diffs = np.abs(preds - targets)
        correct = np.sum(diffs <= tol)
    
    return correct / len(targets) * 100.0

def main():
    """
    End-to-end experiment:
        1. Load Nordland dataset
        2. Unsupervised STDP training
        3. Neuron-to-place assignment
        4. Cross-season VPR evaluation
        5. Metric visualization
    """

    args = _parse_args()

    # --------------------------------------------------------
    # Hyperparameters & Configuration
    # --------------------------------------------------------
    # No auto-scaling of thr/a_plus with resolution.
    # L2 weight normalization + homeostasis handle scale adaptation.
    # Scaling a_plus kills learning at higher dims (update too small to survive L2 renorm).
    default_thr = 1.5
    default_a_plus = 5e-4

    params = {
        # Data paths
        "TRAIN_PATH": [],
        "TEST_PATH": "",
        
        # Training loop
        "EPOCHS": 120,
        "BATCH_SIZE": 64,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "SAVE_RESULTS": True,
        
        # Architecture
        "RESOLUTION": args.resolution,
        "N_IN": args.resolution * args.resolution,
        "N_EXC": args.n_exc,
        
        # Rate Encoder
        "t_steps": 200,
        "rate_scale": 0.25,
        
        # Neuron / STDP Params (Diehl & Cook)
        "beta_e": 0.95,
        "beta_i": 0.90,
        "thr_e_init": args.thr_e_init if args.thr_e_init is not None else default_thr,
        "thr_i": 1.0,
        "a_plus": args.a_plus if args.a_plus is not None else default_a_plus,
        "a_minus": 5e-6,
        "tau_pre": 15.0,
        "tau_post": 15.0,
        "w_max": 1.0,
        "w_ei": 1.0,
        "w_ie": 10.0,
        "thr_eta": args.thr_eta,
        "target_rate": args.target_rate,
        "MAX_SAMPLES": 100,
        "SEED": args.seed,
        "SAVE_MODEL": args.save_model,
        "LOAD_MODEL": args.load_model,
        "DESCRIPTOR": getattr(args, 'descriptor', 'raw'),
    }

    # Set random seeds immediately for reproducibility
    set_seed(params["SEED"])
    print(f"Random seed set to: {params['SEED']}")

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.dataset == "oxford":
        params["TRAIN_PATH"] = [
            os.path.join(ROOT_DIR, "ORC/ORC_sun/2015-08-12-15-04-18/stereo/left"),
            os.path.join(ROOT_DIR, "ORC/ORC_rain/2015-10-29-12-18-17/stereo/left")
        ]
        params["TEST_PATH"] = os.path.join(ROOT_DIR, "ORC/ORC_dusk/2014-11-21-16-07-03/stereo/left")
    elif args.dataset in ["mddrobots", "mddrobots_rooms"]:
        params["MDDROBOTS_ROOT"] = os.path.join(ROOT_DIR, "MDDRobots_dataset")
        params["ROOM"] = args.room
        params["FRAME_SKIP"] = args.frame_skip if args.frame_skip is not None else 4
    else:
        params["TRAIN_PATH"] = [os.path.join(ROOT_DIR, "nordland_clean/data/spring"), os.path.join(ROOT_DIR, "nordland_clean/data/fall")]
        params["TEST_PATH"] = os.path.join(ROOT_DIR, "nordland_clean/data/summer")

    if args.device == "auto":
        params["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        params["DEVICE"] = "cpu"
    else:
        params["DEVICE"] = args.device

    if args.save_results is not None:
        params["SAVE_RESULTS"] = args.save_results

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be a positive integer")
        params["MAX_SAMPLES"] = args.max_samples

    if args.t_steps is not None:
        if args.t_steps <= 0:
            raise ValueError("--t-steps must be a positive integer")
        params["t_steps"] = args.t_steps

    params["WTA_MODE"] = args.wta_mode
    params["ENABLE_HOMEOSTASIS"] = not args.disable_homeostasis
    params["ENABLE_WEIGHT_NORM"] = not args.disable_patch_norm
    params["STANDARD_ASSIGNMENT_ONLY"] = args.standard_assignment_only
    params["SIMULATE_SPILLOVER"] = args.simulate_spillover
    params["SAMPLE_10M"] = args.sample_10m
    params["MAX_SEQ_LEN"] = args.max_seq_len

    if params["SIMULATE_SPILLOVER"]:
        print("Spill-over simulation active: Enforcing BATCH_SIZE=1 to preserve temporal sequence.")
        params["BATCH_SIZE"] = 1

    DEVICE = params["DEVICE"]
    save_results = params.get("SAVE_RESULTS", True)
    print(f"Running on device: {DEVICE}")
    
    # --------------------------------------------------------
    # Results Directory Setup
    # --------------------------------------------------------
    results_dir = None
    weights_dir = None
    stdp_dir = None

    if save_results:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.experiment_name:
            results_dir = os.path.join(ROOT_DIR, f"results/{args.experiment_name}_{timestamp}")
        else:
            results_dir = os.path.join(ROOT_DIR, f"results/results_{timestamp}")
        weights_dir = os.path.join(results_dir, "weights_history")
        stdp_dir = os.path.join(results_dir, "stdp_viz")

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(stdp_dir, exist_ok=True)

        print(f"Saving all results to: {results_dir}")
    else:
        print("SAVE_RESULTS=False: terminal-only mode (no output files, logs, or images).")

    # --------------------------------------------------------
    # Logger Setup (Redirect stdout)
    # --------------------------------------------------------
    if save_results:
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
        if args.dataset == "mddrobots":
            train_loader, test_loader, num_classes = get_mddrobots_loaders(
                dataset_root=params["MDDROBOTS_ROOT"],
                room_name=params["ROOM"],
                batch_size=params["BATCH_SIZE"],
                max_samples=params["MAX_SAMPLES"],
                start_idx=args.start_idx,
                shuffle_train=True,
                frame_skip=params["FRAME_SKIP"],
                resolution=params["RESOLUTION"]
            )
        elif args.dataset == "mddrobots_rooms":
            train_loader, test_loader, num_classes, n_in = get_mddrobots_room_loaders(
                dataset_root=params["MDDROBOTS_ROOT"],
                batch_size=params["BATCH_SIZE"],
                max_samples=params["MAX_SAMPLES"],
                shuffle_train=True,
                frame_skip=params["FRAME_SKIP"],
                resolution=params["RESOLUTION"],
                descriptor=args.descriptor
            )
            params["N_IN"] = n_in
            # Force tolerance to 0 for room classification
            args.tolerance = 0
        else:
            train_loader, test_loader, num_classes = get_nordland_loaders(
                train_path=params["TRAIN_PATH"],
                test_path=params["TEST_PATH"],
                batch_size=params["BATCH_SIZE"],
                max_samples=params["MAX_SAMPLES"],
                start_idx=args.start_idx,
                shuffle_train=True,
                oxford_10m_sampling=params.get("SAMPLE_10M", False),
                resolution=params["RESOLUTION"]
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
        max_samples=params["MAX_SAMPLES"],
        device=DEVICE,
        wta_mode=params["WTA_MODE"],
        enable_homeostasis=params["ENABLE_HOMEOSTASIS"],
        enable_weight_norm=params["ENABLE_WEIGHT_NORM"]
    ).to(DEVICE)

    # --------------------------------------------------------
    # Unsupervised STDP training
    # --------------------------------------------------------
    
    if params["LOAD_MODEL"] and os.path.exists(params["LOAD_MODEL"]):
        print(f"Loading trained model from: {params['LOAD_MODEL']}")
        checkpoint = torch.load(params["LOAD_MODEL"], map_location=DEVICE)
        if "state_dict" in checkpoint:
            net.load_state_dict(checkpoint["state_dict"])
        else:
            net.load_state_dict(checkpoint)
        print("Skipping STDP training.")
    else:
        print("Starting STDP training...")
        net.train()

        total_batches = len(train_loader)
        carried_state = None

        for epoch in range(1, params["EPOCHS"] + 1):
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{params['EPOCHS']}")
                
            for batch_idx, (xb, _) in enumerate(train_loader):
                xb = xb.to(DEVICE)
                spk_in = encoder(xb)
                
                # Monitor only on last batch
                is_last_batch = (batch_idx == total_batches - 1)
                
                if params.get("SIMULATE_SPILLOVER", False):
                    if is_last_batch and save_results:
                        _, history, carried_state = net(spk_in, do_stdp=True, monitor=True, return_state=True, state=carried_state)
                        plot_stdp_monitor(history, epoch, stdp_dir)
                    else:
                        _, carried_state = net(spk_in, do_stdp=True, return_state=True, state=carried_state)
                        
                    # Simulate 150ms Rest Phase (128 steps)
                    REST_STEPS = 128
                    rest_spk_in = torch.zeros((REST_STEPS, xb.size(0), spk_in.size(2)), device=DEVICE)
                    _, carried_state = net(rest_spk_in, do_stdp=True, return_state=True, state=carried_state)
                else:
                    if is_last_batch and save_results:
                        _, history = net(spk_in, do_stdp=True, monitor=True)
                        plot_stdp_monitor(history, epoch, stdp_dir)
                    else:
                        _ = net(spk_in, do_stdp=True)
            
            # Save weights every 10 epochs
            if save_results and epoch % 10 == 0:
                epoch_weight_path = os.path.join(weights_dir, f"epoch_{epoch}_weights.png")
                plot_weights(net.w_in_exc, params["N_EXC"], n_side=params["RESOLUTION"], save_path=epoch_weight_path)
                
    # --------------------------------------------------------
    # Auto-save trained weights to weights/ folder
    # --------------------------------------------------------
    weights_save_dir = os.path.join(ROOT_DIR, "weights")
    os.makedirs(weights_save_dir, exist_ok=True)

    # Build descriptive filename from key network characteristics
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    desc_parts = [
        "vprsnn",
        f"{args.dataset}",
        f"{params['N_EXC']}exc",
        f"{params['N_IN']}in",
        f"{params['EPOCHS']}ep",
        f"{params['t_steps']}ts",
        f"{params['WTA_MODE']}wta",
        f"{'homeo' if params['ENABLE_HOMEOSTASIS'] else 'nohomeo'}",
        f"{'wnorm' if params['ENABLE_WEIGHT_NORM'] else 'nownorm'}",
        f"{params['RESOLUTION']}res",
        f"seed{params['SEED']}",
        timestamp_str,
    ]
    if args.dataset in ["mddrobots", "mddrobots_rooms"]:
        desc_parts.insert(2, params["ROOM"])
    if params.get("DESCRIPTOR", "raw") != "raw":
        desc_parts.insert(2, params["DESCRIPTOR"])
    weights_filename = "_".join(desc_parts) + ".pt"
    weights_save_path = os.path.join(weights_save_dir, weights_filename)

    auto_checkpoint = {
        "state_dict": net.state_dict(),
        "params": {
            "dataset": args.dataset,
            "n_in": params["N_IN"],
            "n_exc": params["N_EXC"],
            "resolution": params["RESOLUTION"],
            "epochs": params["EPOCHS"],
            "t_steps": params["t_steps"],
            "wta_mode": params["WTA_MODE"],
            "enable_homeostasis": params["ENABLE_HOMEOSTASIS"],
            "enable_weight_norm": params["ENABLE_WEIGHT_NORM"],
            "seed": params["SEED"],
            "a_plus": params["a_plus"],
            "a_minus": params["a_minus"],
            "thr_e_init": params["thr_e_init"],
            "target_rate": params["target_rate"],
            "thr_eta": params["thr_eta"],
            "descriptor": params.get("DESCRIPTOR", "raw"),
        },
        "timestamp": timestamp_str,
    }
    if args.dataset in ["mddrobots", "mddrobots_rooms"]:
        auto_checkpoint["params"]["room"] = params["ROOM"]

    torch.save(auto_checkpoint, weights_save_path)
    print(f"Auto-saved trained weights to: {weights_save_path}")

    assignments, avg_rates = get_standard_assignments(
        net=net,
        encoder=encoder,
        loader=train_loader,
        n_exc=params["N_EXC"],
        n_classes=num_classes,
        device=DEVICE
    )

    if params["SAVE_MODEL"]:
        checkpoint = {
            "state_dict": net.state_dict(),
            "assignments": assignments.cpu(),
            "num_classes": num_classes,
            "resolution": params["RESOLUTION"],
            "n_exc": params["N_EXC"]
        }
        if hasattr(train_loader.dataset, "rooms"):
            checkpoint["room_names"] = train_loader.dataset.rooms
            
        torch.save(checkpoint, params["SAVE_MODEL"])
        print(f"Saved trained model and assignments to: {params['SAVE_MODEL']}")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    acc, preds, targets, sim_matrix, prob_matrix, S_Q_cpu, profiler_stats = evaluate_vpr(
        net=net,
        encoder=encoder,
        loader=test_loader,
        assignments=assignments,
        n_classes=num_classes,
        device=DEVICE,
        simulate_spillover=params.get("SIMULATE_SPILLOVER", False),
        tolerance=args.tolerance
    )
    S_Q = S_Q_cpu.to(DEVICE)

    # Save runtime/compute profiling data (FLOPs, latency, power/energy)
    if save_results:
        with open(os.path.join(results_dir, "inference_profile.json"), "w") as f:
            json.dump(profiler_stats, f, indent=4)
        print(f"Saved inference profile to: {os.path.join(results_dir, 'inference_profile.json')}")

    # --------------------------------------------------------
    # Weighted & Probability-based Assignments
    # --------------------------------------------------------
    methods = [
        ("Standard", sim_matrix, preds, acc)
    ]
    
    if not params["STANDARD_ASSIGNMENT_ONLY"]:
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
        
        acc_w = calc_acc(scores_weighted, targets)
        acc_p = calc_acc(scores_prob, targets)
    
        methods.extend([
            ("Weighted", scores_weighted.cpu().numpy(), scores_weighted.argmax(dim=1).cpu().numpy(), acc_w), # Weighted
            ("Weighted+Prob", scores_prob.cpu().numpy(), scores_prob.argmax(dim=1).cpu().numpy(), acc_p)     # Weighted+Prob
        ])
        
        # --------------------------------------------------------
        # 5. Sequential Frame Aggregation (Sliding Window Sweep)
        # --------------------------------------------------------
        print("\n--- Running Sequential Frame Aggregation Sweep ---")
        base_k_values = [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
        k_values = [k for k in base_k_values if k <= params["MAX_SEQ_LEN"]]
        if params["MAX_SEQ_LEN"] not in k_values:
            k_values.append(params["MAX_SEQ_LEN"])
            k_values.sort()
            
        rules = ['product', 'mean']
        
        # We will collect R@100P for plotting
        sweep_results = {'product': [], 'mean': []}
        
        best_k = 5 # default choice for final table
        best_rule = 'product'
        
        for rule in rules:
            print(f"  Aggregation Rule: {rule}")
            for k in k_values:
                agg_scores_t, valid_targets_list, latency_ms = sliding_window_aggregation(
                    scores_prob, targets, k, rule=rule
                )
                
                if len(valid_targets_list) == 0:
                    print(f"    k={k}: Not enough valid frames")
                    sweep_results[rule].append(0.0)
                    continue
                    
                agg_scores_np = agg_scores_t.cpu().numpy()
                valid_targets_np = np.array(valid_targets_list)
                
                r100p = calculate_r_at_100p(agg_scores_np, valid_targets_np, tolerance=args.tolerance)
                p100r = calculate_p_at_100r(agg_scores_np, valid_targets_np, tolerance=args.tolerance)
                acc = calc_acc(agg_scores_t, valid_targets_np, tol=args.tolerance)
                
                print(f"    k={k:2d} (N={len(valid_targets_list)}): R@100P = {r100p:5.2f}% | P@100R = {p100r:5.2f}% | Acc = {acc:5.2f}%")
                
                if k == 5 and rule == "product":
                    print(f"    -> Aggregation latency overhead for k=5 (product): {latency_ms:.4f} ms per query")
                    
                sweep_results[rule].append(r100p)
                
                # Add to methods for full evaluation if it's the baseline (k=1) or the best chosen configuration
                method_name = f"SeqAgg(k={k},{rule})"
                if k == best_k and rule == best_rule:
                    methods.append((method_name, agg_scores_np, agg_scores_t.argmax(dim=1).cpu().numpy(), acc, valid_targets_np))

        # Save sweep plot
        if save_results:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.plot(k_values, sweep_results['product'], marker='o', label='Product Rule')
            plt.plot(k_values, sweep_results['mean'], marker='s', label='Mean Rule')
            plt.xlabel('Window Size (k)')
            plt.ylabel('R@100P (%)')
            plt.title('Sequential Aggregation Performance')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(results_dir, "aggregation_sweep_r100p.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"\nSaved aggregation sweep plot to {plot_path}")
            
        # --------------------------------------------------------
        # 5.b Velocity Sensitivity Sweep (Ablation)
        # --------------------------------------------------------
        print("\n--- Running Velocity Sensitivity Sweep (Product Rule) ---")
        velocity_factors = [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0]
        for v_k in [5, 15]:
            print(f"  k = {v_k}:")
            for v in velocity_factors:
                agg_scores_t, valid_targets_list, _ = sliding_window_aggregation(
                    scores_prob, targets, k=v_k, rule="product", velocity_factor=v
                )
                if len(valid_targets_list) > 0:
                    agg_scores_np = agg_scores_t.cpu().numpy()
                    valid_targets_np = np.array(valid_targets_list)
                    r100p = calculate_r_at_100p(agg_scores_np, valid_targets_np)
                    print(f"    v={v:4.2f}x: R@100P = {r100p:5.2f}%")
                else:
                    print(f"    v={v:4.2f}x: Not enough valid frames")
            
    # 6. Unified Result Processing & Plotting
    # --------------------------------------------------------
    # Handle ConcatDataset for training (use first dataset as reference for viz)
    vis_train_ds = train_loader.dataset
    if isinstance(vis_train_ds, torch.utils.data.ConcatDataset):
        vis_train_ds = vis_train_ds.datasets[0]
    
    # Store metrics for final table
    final_metrics = []
    
    # Store raw predictions for GPS/trajectory visualization
    predictions_export = {
        "image_files": list(test_loader.dataset.image_files),
        "methods": {}
    }

    for method_tuple in methods:
        if len(method_tuple) == 4:
            name, scores, pred_labels, accuracy = method_tuple
            curr_targets = targets
        else:
            name, scores, pred_labels, accuracy, curr_targets = method_tuple

        predictions_export["methods"][name] = {
            "pred_labels": pred_labels.tolist() if hasattr(pred_labels, "tolist") else list(pred_labels),
            "true_labels": curr_targets.tolist() if hasattr(curr_targets, "tolist") else list(curr_targets)
        }

        print(f"\nProcessing Results for: {name} (N={len(curr_targets)})...")
        
        # Create Method Directory
        method_dir = None
        if save_results:
            method_dir = os.path.join(results_dir, name.replace(" ", "_").replace("(","").replace(")","").replace(",","_"))
            os.makedirs(method_dir, exist_ok=True)
        
        # Calculate Metrics
        p100r = calculate_p_at_100r(scores, curr_targets)
        r100p = calculate_r_at_100p(scores, curr_targets)
        auc_pr = calculate_auc_pr(scores, curr_targets)
        
        final_metrics.append({
            "Method": name,
            "Accuracy": accuracy,
            "P@100R": p100r,
            "R@100P": r100p,
            "AUC-PR": auc_pr,
            "N": len(curr_targets)
        })
        
        if save_results:
            # Visualizations
            # 1. Distance Matrix
            plot_distance_matrix(scores, save_path=os.path.join(method_dir, "distance_matrix.png"), targets=curr_targets)

            # 2. PR Curve
            plot_pr_curve(scores, curr_targets, num_classes, save_path=os.path.join(method_dir, "pr_curve.png"))

            # 3. Recall @ N
            plot_recall_at_n(scores, curr_targets, save_path=os.path.join(method_dir, "recall_at_n.png"))

            # 4. Qualitative Results (Reuse Standard Assignments for neuron interpretation, but use Method Preds)
            # Note: Assignments are static (Standard Training), but Preds change per method.
            visualize_qualitative_results(
                query_ds=test_loader.dataset,
                database_ds=vis_train_ds,
                assignments=assignments,
                weights=net.w_in_exc,
                preds=pred_labels,
                targets=curr_targets,
                save_path=os.path.join(method_dir, "qualitative.png")
            )
    
    # --------------------------------------------------------
    # 7. Final Summary Table
    # --------------------------------------------------------
    print("\n" + "="*95)
    
    header = f"{'METHOD':<30} | {'ACCURACY':<10} | {'P@100R':<12} | {'R@100P':<10} | {'AUC-PR':<10} | {'N':<5}"
    print(header)
    print("-" * 105)
    
    for m in final_metrics:
        print(f"{m['Method']:<30} | {m['Accuracy']:<9.2f}% | {m['P@100R']:<11.2f}% | {m['R@100P']:<9.2f}% | {m['AUC-PR']:<9.2f}% | {m['N']:<5}")
             
    print("="*105 + "\n")

    if save_results:
        preds_path = os.path.join(results_dir, "predictions.json")
        with open(preds_path, 'w') as f:
            json.dump(predictions_export, f)
        print(f"Saved raw predictions for mapping to {preds_path}")


if __name__ == "__main__":
    main()
