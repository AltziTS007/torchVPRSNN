"""
vprsnn_evaluation.py

Evaluation, calibration, and dataset utilities for
Spiking Neural Network–based Visual Place Recognition (VPR).

This file implements:
- Neuron-to-place assignment strategies
- VPR inference rules (Standard + WNA)
- Retrieval-style metrics (R@100P, P@100R)
- Visualization utilities
- Nordland dataset loading and preprocessing
- MDDRobots indoor dataset loading and preprocessing

IMPORTANT:
This file assumes:
- One-to-one correspondence between reference/query images
  (Spring/Fall for Nordland; train/test traversals for MDDRobots)
- Single ground-truth place per query
- No temporal alignment (pure appearance-based VPR)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import cv2
import glob
import os
import subprocess
import time
import metrics as vpr_metrics


# ============================================================
# 1. NEURON-TO-PLACE ASSIGNMENT
# ============================================================

def get_standard_assignments(net, encoder, loader, n_exc, n_classes, device="cpu"):
    """
    Standard Assignment (SA) calibration phase.

    Each excitatory neuron is assigned to the place (class)
    for which it exhibits the highest average firing rate.

    This is the classical Diehl & Cook / MNIST-style assignment,
    adapted to VPR where "class" = physical place.

    Procedure:
        For each training image:
            - Encode image → spike trains
            - Run SNN forward pass (plasticity OFF)
            - Count spikes per excitatory neuron

        For each neuron:
            - Compute mean firing rate per class
            - Assign neuron to argmax class
            - If neuron never fires → unassigned (-1)

    Args:
        net: Trained SNN (DiehlCookVPRSNN)
        encoder: Spike encoder (RateEncoder)
        loader: DataLoader for calibration set
        n_exc (int): Number of excitatory neurons
        n_classes (int): Number of places
        device (str): cpu / cuda

    Returns:
        assignments (Tensor): [n_exc]
            assignments[i] = class index or -1 if inactive

        avg_rates (Tensor): [n_exc, n_classes]
            Average firing rate per neuron per class
    """
    net.eval()

    # Total spike counts per neuron per class
    total_spikes_per_class = torch.zeros(n_exc, n_classes, device=device)
    img_counts_per_class = torch.zeros(n_classes, device=device)

    print("Calibrating (Get Assignments - Standard)...")
    with torch.no_grad():
        for xb, yb in tqdm(loader):
            xb = xb.to(device)
            yb = yb.to(device)

            spk_in = encoder(xb)

            # Sum over time → [Batch, n_exc]
            exc_counts = net(spk_in, do_stdp=False).sum(dim=0)

            # Accumulate per class
            for i in range(len(yb)):
                label = yb[i].item()
                total_spikes_per_class[:, label] += exc_counts[i]
                img_counts_per_class[label] += 1

    # Average firing rate per class
    valid_classes = img_counts_per_class > 0
    avg_rates = torch.zeros_like(total_spikes_per_class)
    avg_rates[:, valid_classes] = (
        total_spikes_per_class[:, valid_classes]
        / img_counts_per_class[valid_classes]
    )

    # Assign neuron to class with maximum rate
    max_rates, assignments = avg_rates.max(dim=1)

    # Neurons that never fired remain unassigned
    assignments[max_rates == 0] = -1

    return assignments, avg_rates

# ============================================================
# 2. STANDARD VPR EVALUATION
# ============================================================

def evaluate_vpr(net, encoder, loader, assignments, n_classes, device='cpu', 
                 simulate_spillover=False, tolerance=0):
    """
    Evaluates VPR using Standard Assignment inference.

    Inference rule for a query image:
        1. Obtain spike count vector r [n_exc]
        2. For each place c:
            - Select neurons assigned to c
            - Score(c) = mean spike count of those neurons
        3. Predict argmax_c Score(c)

    This is a *neuron-voting* mechanism.

    Returns:
        accuracy (%)
        predictions list
        targets list
        similarity_matrix [N_test, N_places]
        prob_matrix (normalized similarity)
    """
    net.eval()

    # Keep evaluation responsive by default. Enable profiler-based FLOPs only if requested.
    flops_mode = os.environ.get("VPRSNN_FLOPS_MODE", "estimate").strip().lower()
    use_profiler_flops = (flops_mode == "profiler")

    def _estimate_flops_fallback(model, sample_spk):
        """
        Approximate dense FLOPs for one forward pass.
        This is used only when profiler-based FLOP counting is unavailable.
        """
        t_steps = sample_spk.size(0)
        n_in = model.n_in
        n_exc = model.n_exc

        # 3 dense matmuls per timestep in forward:
        # pre@W_in_exc, spk_e@W_exc_inh, spk_i@W_inh_exc
        # Approximate MAC as 2 FLOPs (mul + add).
        flops_per_t = 2.0 * (
            (n_in * n_exc) +
            (n_exc * n_exc) +
            (n_exc * n_exc)
        )
        return float(flops_per_t * t_steps)

    def _measure_flops_one_query(model, sample_spk):
        """
        Measures FLOPs for one query sample.
        Returns (flops, source).
        """
        if not use_profiler_flops:
            return _estimate_flops_fallback(model, sample_spk), "analytic_estimate"

        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if sample_spk.is_cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            with torch.no_grad():
                if sample_spk.is_cuda:
                    torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=activities,
                    with_flops=True,
                    record_shapes=False,
                    profile_memory=False,
                ) as prof:
                    _ = model(sample_spk, do_stdp=False)
                if sample_spk.is_cuda:
                    torch.cuda.synchronize()

            flops = 0.0
            for evt in prof.key_averages():
                evt_flops = getattr(evt, "flops", 0)
                if evt_flops is not None:
                    flops += float(evt_flops)

            if flops > 0:
                return flops, "torch.profiler"
        except Exception:
            pass

        return _estimate_flops_fallback(model, sample_spk), "analytic_estimate"

    def _read_rapl_energy_uj():
        """Reads total package energy from Linux RAPL (microjoules)."""
        energy_paths = glob.glob("/sys/class/powercap/*/energy_uj")
        energy_paths += glob.glob("/sys/class/powercap/*/*/energy_uj")
        if not energy_paths:
            return None

        total_uj = 0
        found = False
        for path in energy_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    total_uj += int(f.read().strip())
                    found = True
            except (OSError, ValueError):
                continue

        return total_uj if found else None

    def _read_gpu_power_w():
        """Reads instantaneous GPU power via nvidia-smi (watts)."""
        if os.environ.get("VPRSNN_ENABLE_NVIDIA_SMI", "1").strip() in {"0", "false", "False"}:
            return None

        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            values = []
            for line in proc.stdout.strip().splitlines():
                line = line.strip()
                if line:
                    values.append(float(line))
            if not values:
                return None
            return float(np.mean(values))
        except Exception:
            return None

    t_start = None
    rapl_start_uj = None
    gpu_power_start_w = None

    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    similarity_matrix = []

    print("Evaluating VPR...")
    if use_profiler_flops:
        print("FLOPs mode: torch.profiler (may be slower)")
    else:
        print("FLOPs mode: analytic_estimate (set VPRSNN_FLOPS_MODE=profiler to enable profiler)")
    all_spike_counts_list = []
    measured_flops = None
    flops_source = "unavailable"
    
    carried_state = None
    carried_rest_spikes = None
    
    with torch.no_grad():
        for xb, yb in tqdm(loader):
            xb = xb.to(device)
            yb = yb.to(device)

            spk_in = encoder(xb)

            if measured_flops is None and spk_in.size(1) > 0:
                one_query_spk = spk_in[:, :1, :]
                measured_flops, flops_source = _measure_flops_one_query(net, one_query_spk)

            if t_start is None:
                t_start = time.perf_counter()
                rapl_start_uj = _read_rapl_energy_uj()
                gpu_power_start_w = _read_gpu_power_w()

            if simulate_spillover:
                # Sliding Window's ODE proxy: T=300 (350ms). Rest=150ms => ~128 timesteps
                REST_STEPS = 128
                exc_counts_list = []
                for b_idx in range(xb.size(0)):
                    spk_in_b = spk_in[:, b_idx:b_idx+1, :]
                    
                    # 1. Simulate Image N
                    spikes_b, carried_state = net(spk_in_b, do_stdp=False, return_state=True, state=carried_state)
                    spikes_b_sum = spikes_b.sum(dim=0)
                    
                    # 2. Add spikes fired during the PREVIOUS resting phase (Sliding Window spill-over bug)
                    if carried_rest_spikes is not None:
                        total_spikes = spikes_b_sum + carried_rest_spikes
                    else:
                        total_spikes = spikes_b_sum
                        
                    exc_counts_list.append(total_spikes)
                    
                    # 3. Simulate Resting Phase for Image N
                    rest_spk_in = torch.zeros((REST_STEPS, 1, spk_in.size(2)), device=spk_in.device)
                    rest_spikes, carried_state = net(rest_spk_in, do_stdp=False, return_state=True, state=carried_state)
                    carried_rest_spikes = rest_spikes.sum(dim=0)
                    
                exc_counts = torch.cat(exc_counts_list, dim=0)
            else:
                exc_counts = net(spk_in, do_stdp=False).sum(dim=0)

            for b in range(xb.size(0)):
                sample_counts = exc_counts[b]
                scores = torch.zeros(n_classes, device=device)

                # Mask inactive neurons
                valid_mask = (assignments != -1)
                valid_assignments = assignments[valid_mask]
                valid_counts = sample_counts[valid_mask]

                # Aggregate mean firing per class
                for c in torch.unique(valid_assignments):
                    c = c.item()
                    class_mask = (valid_assignments == c)
                    scores[c] = valid_counts[class_mask].mean()

                similarity_matrix.append(scores.cpu().numpy())

                pred = scores.argmax().item()
                target = yb[b].item()

                correct += int(abs(pred - target) <= tolerance)
                total += 1

                all_preds.append(pred)
                all_targets.append(target)
            
            # Store batch spike counts
            all_spike_counts_list.append(exc_counts.cpu()) # Move to CPU to save GPU memory

    accuracy = 100 * correct / max(total, 1)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    if t_start is None:
        t_start = time.perf_counter()
    t_end = time.perf_counter()
    elapsed_s = t_end - t_start

    rapl_end_uj = _read_rapl_energy_uj()
    gpu_power_end_w = _read_gpu_power_w()

    avg_power_w = None
    energy_wh = None
    energy_source = "unavailable"

    if rapl_start_uj is not None and rapl_end_uj is not None and rapl_end_uj >= rapl_start_uj:
        delta_uj = rapl_end_uj - rapl_start_uj
        delta_j = delta_uj / 1e6
        energy_wh = delta_j / 3600.0
        avg_power_w = delta_j / max(elapsed_s, 1e-9)
        energy_source = "linux_rapl"
    elif gpu_power_start_w is not None and gpu_power_end_w is not None:
        avg_power_w = 0.5 * (gpu_power_start_w + gpu_power_end_w)
        energy_wh = avg_power_w * (elapsed_s / 3600.0)
        energy_source = "nvidia_smi_snapshot"

    effective_energy_per_inference_j = None
    if energy_wh is not None and total > 0:
        effective_energy_per_inference_j = float((energy_wh * 3600.0) / total)

    profiler_stats = {
        "inference_total_time_s": float(elapsed_s),
        "inference_time_per_query_ms": float((elapsed_s / max(total, 1)) * 1000.0),
        "num_queries": int(total),
        "flops_per_query": float(measured_flops) if measured_flops is not None else None,
        "flops_source": flops_source,
        "avg_power_w": float(avg_power_w) if avg_power_w is not None else None,
        "energy_wh": float(energy_wh) if energy_wh is not None else None,
        "effective_energy_per_inference_j": effective_energy_per_inference_j,
        "energy_source": energy_source,
    }

    print(f"Accuracy: {accuracy:.2f}%")
    print("\n--- Inference Profiling ---")
    if profiler_stats["flops_per_query"] is not None:
        print(f"FLOPs/query: {profiler_stats['flops_per_query']:.2f} (source: {profiler_stats['flops_source']})")
    else:
        print("FLOPs/query: unavailable")
    print(f"Total inference time: {profiler_stats['inference_total_time_s']:.4f} s")
    print(f"Inference time/query: {profiler_stats['inference_time_per_query_ms']:.4f} ms")
    if profiler_stats["energy_wh"] is not None:
        print(f"Energy: {profiler_stats['energy_wh']:.6f} Wh")
        if profiler_stats["effective_energy_per_inference_j"] is not None:
            print(f"Effective Energy per inference: {profiler_stats['effective_energy_per_inference_j']:.6f} J")
        print(f"Average Power: {profiler_stats['avg_power_w']:.4f} W")
        print(f"Energy source: {profiler_stats['energy_source']}")
    else:
        print("Energy/Power: unavailable (RAPL and nvidia-smi not accessible)")
    
    # Stack all spike counts
    if all_spike_counts_list:
        all_spike_counts = torch.cat(all_spike_counts_list, dim=0).to(device)
    else:
        all_spike_counts = torch.empty(0, net.n_exc, device=device)

    similarity_matrix = np.array(similarity_matrix)

    # Retrieval metric (raw)
    r_at_100p = calculate_r_at_100p(similarity_matrix, np.array(all_targets))
    print(f"R@100P (Raw): {r_at_100p:.2f}%")

    # --------------------------------------------------------
    # Probability normalization (min-max + row normalization)
    # --------------------------------------------------------
    sim_min = similarity_matrix.min(axis=1, keepdims=True)
    sim_max = similarity_matrix.max(axis=1, keepdims=True)
    sim_range = sim_max - sim_min
    sim_range[sim_range == 0] = 1.0

    sim_norm = (similarity_matrix - sim_min) / sim_range
    row_sums = sim_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    prob_matrix = sim_norm / row_sums

    print("\n--- Probability-Based Metrics ---")
    p_at_100r_prob = calculate_p_at_100r(prob_matrix, np.array(all_targets))
    r_at_100p_prob = calculate_r_at_100p(prob_matrix, np.array(all_targets))
    print("--------------------------------")

    # Stack all spike counts to form S_Q [n_query, n_exc]
    
    return accuracy, all_preds, all_targets, similarity_matrix, prob_matrix, all_spike_counts, profiler_stats

# ============================================================
# 4. PLOTTING FUNCTIONS (Paper Replication)
# ============================================================

from sklearn.metrics import precision_recall_curve, auc

def _prepare_metrics_data(similarity_matrix, targets, tolerance=0):
    """
    Prepares data for metrics.py functions.
    metrics.py expects:
        S (Similarity): [N_ref, N_query] (cols are queries)
        GT (Ground Truth): [N_ref, N_query] (cols are queries)
    
    Our similarity_matrix is [N_query, N_ref].
    So we need to Transpose.
    """
    # S_in: [N_ref, N_query]
    S_in = similarity_matrix.T
    
    n_queries = similarity_matrix.shape[0]
    n_refs = similarity_matrix.shape[1]
    
    # GThard: [N_ref, N_query]
    GThard = np.zeros((n_refs, n_queries), dtype=int)
    
    for i, target in enumerate(targets):
        if target < n_refs:
            start = max(0, target - tolerance)
            end = min(n_refs, target + tolerance + 1)
            GThard[start:end, i] = 1
            
    return S_in, GThard

def plot_distance_matrix(similarity_matrix, save_path="vpr_distance_matrix.png", targets=None):
    """
    Plots the similarity matrix (Scores).
    X-axis: Template (Place-Match)
    Y-axis: Test Image (Query)
    """
    plt.figure(figsize=(10, 8))
    # Normalize for better visualization
    if similarity_matrix.max() > 0:
        sim_norm = similarity_matrix / similarity_matrix.max()
    
    # If targets provided, sort rows by target label to visualize diagonal
    if targets is not None:
        sorted_indices = np.argsort(targets)
        similarity_matrix = similarity_matrix[sorted_indices, :]
        ylabel = "Test Image Index (Sorted by GT Label)"
    else:
        ylabel = "Test Image Index"
        
    sns.heatmap(similarity_matrix, cmap='viridis')
    plt.title("Cosine Similarity Matrix (Test vs Train Neurons)")
    plt.xlabel("Train Place Index")
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Distance Matrix to {save_path}")
    
def calculate_auc_pr(similarity_matrix, targets):
    S_in, GThard = _prepare_metrics_data(similarity_matrix, targets)
    P, R = vpr_metrics.createPR(S_in, GThard, matching='single', n_thresh=100)
    return auc(R, P) * 100.0

def plot_pr_curve(similarity_matrix, targets, n_classes, save_path="pr_curve.png"):
    """
    Computes and plots Precision-Recall curve using metrics.py
    """
    S_in, GThard = _prepare_metrics_data(similarity_matrix, targets)
    
    # createPR returns lists P, R
    P, R = vpr_metrics.createPR(S_in, GThard, matching='single', n_thresh=100)
    
    # Calculate AUC
    pr_auc = auc(R, P)
    
    plt.figure(figsize=(8, 6))
    plt.plot(R, P, lw=2, label=f'AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (metrics.py)')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved PR Curve to {save_path}")
def plot_recall_at_n(similarity_matrix, targets, n_values=[1, 5, 10, 20], save_path="recall_at_n.png"):
    """
    Computes Recall@N using metrics.py recallAtK
    """
    recalls = []
    
    S_in, GThard = _prepare_metrics_data(similarity_matrix, targets)
    
    for n in n_values:
        # recallAtK expects S and GT. It does NOT support matching='single' 
        req_recall = vpr_metrics.recallAtK(S_in, GThard, K=n)
        recalls.append(req_recall * 100.0) # Convert to percentage
        print(f"Recall@{n}: {req_recall * 100.0:.2f}%")
        
    plt.figure(figsize=(8, 6))
    plt.plot(n_values, recalls, marker='o', linestyle='-', lw=2)
    plt.xlabel('N (Top Candidates)')
    plt.ylabel('Recall (%)')
    plt.title('Recall @ N')
    plt.xticks(n_values)
    plt.ylim(0, 105)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Recall@N plot to {save_path}")

def plot_weights(weights, n_exc, n_side=28, n_rows=20, n_cols=20, save_path="weights.png"):
    """
    Plots the receptive fields (weights) of the excitatory neurons.
    Weights are expected to be [n_in, n_exc].
    Handles 2D square image inputs (e.g. 28x28) and 1D feature descriptors (e.g. Spatial Pyramid 380-D).
    """
    w = weights.cpu().detach().numpy()
    n_in = w.shape[0]

    num_to_plot = min(n_exc, 400)
    n_cols = int(np.ceil(np.sqrt(num_to_plot)))
    n_rows = int(np.ceil(num_to_plot / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))

    for i in range(n_rows * n_cols):
        ax = axes.flat[i] if hasattr(axes, 'flat') else axes
        if i < num_to_plot and i < n_exc:
            weight_vec = w[:, i]
            if n_in == n_side * n_side:
                img = weight_vec.reshape(n_side, n_side)
            else:
                side = int(np.sqrt(n_in))
                if side * side == n_in:
                    img = weight_vec.reshape(side, side)
                else:
                    img = weight_vec.reshape(1, -1)
            ax.imshow(img, cmap='hot', interpolation='nearest', aspect='auto')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_neuron_assignments(assignments, avg_rates, n_classes, save_path="neuron_assignments.png"):
    """
    Plots visualizations of neuron-to-place assignments:
    1. Histogram of neurons per place.
    2. Heatmap of average firing rates (Neurons vs Places).
    """
    assignments_np = assignments.cpu().numpy()
    
    # 1. Histogram of assignments
    plt.figure(figsize=(10, 5))
    plt.hist(assignments_np, bins=range(-1, n_classes + 1), rwidth=0.8, align='left')
    plt.xlabel("Place ID (-1 = Unassigned)")
    plt.ylabel("Number of Neurons")
    plt.title("Neuron Assignments per Place")
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(save_path.replace(".png", "_hist.png"))
    plt.close()
    print(f"Saved Assignment Histogram to {save_path.replace('.png', '_hist.png')}")
    
    # 2. Firing Rate Heatmap
    # avg_rates: [n_exc, n_classes]
    # We sort neurons by their assigned class for better visualization
    avg_rates_np = avg_rates.cpu().numpy()
    
    # Sort indices: unassigned (-1) first, then by class
    sorted_indices = np.argsort(assignments_np)
    sorted_rates = avg_rates_np[sorted_indices]
    
    plt.figure(figsize=(12, 8))
    # Normalize for visibility
    if sorted_rates.max() > 0:
        sorted_rates_norm = sorted_rates / sorted_rates.max()
    else:
        sorted_rates_norm = sorted_rates
        
    plt.imshow(sorted_rates_norm, interpolation='nearest', cmap='viridis', aspect='auto')
    plt.colorbar(label="Normalized Firing Rate")
    plt.xlabel("Place Label (Class)")
    plt.ylabel("Neuron Index (Sorted by Assignment)")
    plt.title("Neuron Firing Rates (Sorted by Assignment)")
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_heatmap.png"))
    plt.close()
    print(f"Saved Firing Rate Heatmap to {save_path.replace('.png', '_heatmap.png')}")

def calculate_r_at_100p(similarity_matrix, targets, tolerance=0):
    """
    Calculates Recall at 100% Precision using metrics.py
    """
    S_in, GThard = _prepare_metrics_data(similarity_matrix, targets, tolerance=tolerance)
    
    # recallAt100precision returns a float (0.0 to 1.0)
    r100p = vpr_metrics.recallAt100precision(S_in, GThard, matching='single')
    
    return r100p * 100.0

def calculate_p_at_100r(similarity_matrix, targets, tolerance=0):
    """
    Calculates Precision at 100% Recall using metrics.py createPR
    """
    S_in, GThard = _prepare_metrics_data(similarity_matrix, targets, tolerance=tolerance)
    
    P, R = vpr_metrics.createPR(S_in, GThard, matching='single', n_thresh=200)
    P = np.array(P)
    R = np.array(R)
    
    # We want P where R is max (1.0 ideally)
    max_recall = R.max()
    
    # Get P at max recall
    # If multiple, take the one at the loosest threshold (last one)?
    # Usually PR curve: as R increases, P decreases.
    # We want P when R is 1.0. If R never reaches 1.0, we take P at max R.
    
    # Filter where R == max_recall
    p_at_max_r = P[R == max_recall]
    
    # Use the last one (assuming sorted by threshold, which createPR does: startV (max score) to endV (min score))
    # createPR iterates thresholds from max to min.
    # High thresh -> Low Recall, High Precision.
    # Low thresh -> High Recall, Low Precision.
    # So the *last* elements in P, R correspond to Low Threshold (High Recall).
    # So we want the last element where R == max_recall.
    
    val = p_at_max_r[-1] * 100.0
    print(f"Max Recall: {max_recall*100:.2f}%, P@MaxR: {val:.2f}%")
    
    return val

def visualize_qualitative_results(query_ds, database_ds, assignments, weights, preds, targets, save_path="qualitative_results.png"):
    """
    Visualizes qualitative results (Correct vs Incorrect matches).
    Now uses standard VPR terminology: Query vs Database (Reference).
    Handles shuffled test data by using `targets` (Place ID) to index the Query dataset.
    """
    weights_np = weights.cpu().detach().numpy()
    assignments_np = assignments.cpu().numpy()
    
    # helper to check image path
    def check_image_path(ds, idx):
        if hasattr(ds, 'image_files') and len(ds.image_files) > idx:
            return os.path.basename(ds.image_files[idx])
        return f"Idx {idx}"

    # helper to get raw image
    def get_raw_image(ds, idx):
        if hasattr(ds, 'samples'):
            path = ds.samples[idx][0]
        else:
            path = os.path.join(ds.dir_path, ds.image_files[idx])
        return Image.open(path).convert("RGB")
        
    # Find indices
    correct_indices = [i for i, (p, t) in enumerate(zip(preds, targets)) if p == t]
    incorrect_indices = [i for i, (p, t) in enumerate(zip(preds, targets)) if p != t]
    
    # Select examples to plot
    examples = []
    if correct_indices:
        examples.append((correct_indices[0], "Correct Match"))
    if incorrect_indices:
        examples.append((incorrect_indices[0], "Incorrect Match"))
        
    if not examples:
        print("No examples found for visualization.")
        return

    for i, (idx, title) in enumerate(examples):
        # Target Place ID (The Ground Truth Label)
        place_id = targets[idx]
        # Predicted Place ID (The ID retrieved from Database)
        pred_place_id = preds[idx]
        
        # Get Images
        # Query (Test Set) - IMPORTANT: Use `place_id` if Test set is shuffled and 1-to-1 matching!
        # Assumption: NordlandDataset(index) == Place(index). So if target is 42, we want image index 42 from Query DS.
        # Original code used `idx` (batch index), which is wrong if shuffled.
        query_idx = place_id
        
        query_tensor, _ = query_ds[query_idx]
        query_proc = query_tensor.squeeze().cpu().numpy()
        query_raw = get_raw_image(query_ds, query_idx)
        query_fname = check_image_path(query_ds, query_idx)
        
        # Ground Truth Reference (Database Set) - The Correct Match
        gt_tensor, _ = database_ds[place_id]
        gt_proc = gt_tensor.squeeze().cpu().numpy()
        gt_raw = get_raw_image(database_ds, place_id)
        gt_fname = check_image_path(database_ds, place_id)
        
        # Retrieved Reference (Database Set) - The Model's Prediction
        retrieved_tensor, _ = database_ds[pred_place_id]
        retrieved_proc = retrieved_tensor.squeeze().cpu().numpy()
        retrieved_raw = get_raw_image(database_ds, pred_place_id)
        retrieved_fname = check_image_path(database_ds, pred_place_id)
        
        # Neurons assigned to the RETRIEVED place (to see why it fired)
        assigned_neurons = np.where(assignments_np == pred_place_id)[0]
        n_assigned = len(assigned_neurons)
        
        # Pick top 3 neurons (or fewer)
        neurons_to_show = assigned_neurons[:3]
        
        # Setup Grid
        n_neurons_show = len(neurons_to_show)
        n_cols = max(3, n_neurons_show) 
        n_rows = 3
        
        fig = plt.figure(figsize=(4 * n_cols, 10))
        plt.suptitle(f"{title}: Query P{place_id} -> Retrieved P{pred_place_id}\n({n_assigned} neurons assigned to P{pred_place_id})", fontsize=16)
        
        # Row 1: Proc
        if query_proc.ndim == 1:
            query_proc = query_proc.reshape(1, -1)
        if retrieved_proc.ndim == 1:
            retrieved_proc = retrieved_proc.reshape(1, -1)
        if gt_proc.ndim == 1:
            gt_proc = gt_proc.reshape(1, -1)

        # Query
        ax = plt.subplot(n_rows, n_cols, 1)
        ax.imshow(query_proc, cmap='gray', aspect='auto')
        ax.set_title(f"Query (Input)\n{query_fname}")
        ax.axis('off')

        # Retrieved (Database)
        ax = plt.subplot(n_rows, n_cols, 2)
        ax.imshow(retrieved_proc, cmap='gray', aspect='auto')
        ax.set_title(f"Retrieved (Database)\n{retrieved_fname}")
        ax.axis('off')
        
        # Ground Truth (Database)
        ax = plt.subplot(n_rows, n_cols, 3)
        ax.imshow(gt_proc, cmap='gray', aspect='auto')
        ax.set_title(f"Ground Truth (Database)\n{gt_fname}")
        ax.axis('off')
        
        # Row 2: Raw
        # Query
        ax = plt.subplot(n_rows, n_cols, n_cols + 1)
        ax.imshow(query_raw)
        ax.set_title(f"Query (Raw)\n{query_fname}")
        ax.axis('off')

        # Retrieved
        ax = plt.subplot(n_rows, n_cols, n_cols + 2)
        ax.imshow(retrieved_raw)
        ax.set_title(f"Retrieved (Raw)\n{retrieved_fname}")
        ax.axis('off')
        
        # Ground Truth
        ax = plt.subplot(n_rows, n_cols, n_cols + 3)
        ax.imshow(gt_raw)
        ax.set_title(f"Ground Truth (Raw)\n{gt_fname}")
        ax.axis('off')
        
        # Row 3: Neurons
        if n_neurons_show > 0:
            n_in = weights_np.shape[0]
            side = int(np.sqrt(n_in))
            for k, nid in enumerate(neurons_to_show):
                ax = plt.subplot(n_rows, n_cols, 2*n_cols + k + 1)
                if side * side == n_in:
                    w_img = weights_np[:, nid].reshape(side, side)
                else:
                    w_img = weights_np[:, nid].reshape(1, -1)
                ax.imshow(w_img, cmap='hot', aspect='auto')
                ax.set_title(f"Neuron {nid}")
                ax.axis('off')
        else:
             plt.text(0.5, 0.1, "No neurons assigned to this class", ha='center')

        plt.tight_layout()
        out_name = save_path.replace(".png", f"_{title.split()[0].lower()}.png")
        plt.savefig(out_name)
        plt.close()
        print(f"Saved qualitative result to {out_name}")
# ============================================================
# 5. DATASET HELPERS (Nordland Custom)
# ============================================================
from torch.utils.data import Dataset
from PIL import Image
import os
import seaborn as sns
class NordlandDataset(Dataset):
    """
    Custom Dataset for Nordland that assumes images are sequential.
    - file_001.png in Spring corresponds to Place 0
    - file_001.png in Fall corresponds to Place 0
    
    It reads a flat directory of images, sorts them by filename, 
    and assigns the index as the label.
    """
    def __init__(self, dir_path, transform=None, valid_files=None):
        self.dir_path = dir_path
        self.transform = transform
        
        # extensions to look for
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        
        # Get all image files and sort them to ensure alignment
        files = [
            f for f in os.listdir(dir_path) 
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        
        if valid_files is not None:
            files = [f for f in files if f in valid_files]
            
        self.image_files = sorted(files)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {dir_path}")
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.dir_path, img_name)
        
        image = Image.open(img_path).convert("RGB") # Convert to RGB first
        
        if self.transform:
            image = self.transform(image)
            
        # Label is strictly the index (Place ID)
        label = idx
        
        return image, label

class PatchNormalization:
    """
    Applies Patch Normalization to a tensor image.
    Splits the image into patches and Min-Max normalizes each patch.
    """
    def __init__(self, patch_size=7):
        self.patch_size = patch_size

    def __call__(self, img_tensor):
        # img_tensor shape: [C, H, W] (e.g., [1, 28, 28])
        # We assume grayscale [1, H, W]
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0)
            
        C, H, W = img_tensor.shape
        
        # Unfold to patches
        # [C, H, W] -> [1, C, H, W] for unfold
        patches = img_tensor.unsqueeze(0).unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches shape: [1, C, H_steps, W_steps, patch_h, patch_w]
        
        # Normalize each patch
        # Flatten the last two dimensions for min/max
        # [1, C, H_steps, W_steps, patch_size*patch_size]
        patches_flat = patches.contiguous().view(*patches.shape[:4], -1)
        
        min_vals = patches_flat.min(dim=-1, keepdim=True)[0]
        max_vals = patches_flat.max(dim=-1, keepdim=True)[0]
        
        # Avoid div by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        patches_norm_flat = (patches_flat - min_vals) / range_vals
        
        # Reshape back to patches
        patches_norm = patches_norm_flat.view(*patches.shape)
        
        # Fold back (Reassemble)
        # Since patches are non-overlapping, we can just permute and reshape
        # patches: [1, C, H_steps, W_steps, ph, pw]
        # target: [1, C, H, W]
        # Permute to: [1, C, H_steps, ph, W_steps, pw]
        patches_norm = patches_norm.permute(0, 1, 2, 4, 3, 5)
        
        # Reshape to [1, C, H, W]
        img_reconstructed = patches_norm.contiguous().view(1, C, H, W)
        
        img_reconstructed = patches_norm.contiguous().view(1, C, H, W)
        
        return img_reconstructed.squeeze(0)

class CLAHE:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Operates on PIL Image or Tensor (converts to numpy, applies, converts back).
    """
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        # Expecting PIL Image
        img_np = np.array(img)
        
        # Ensure grayscale
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
        # Apply CLAHE
        img_clahe = self.clahe.apply(img_np)
        
        # Convert back to PIL
        return Image.fromarray(img_clahe)


class DoGTransform:
    """
    Difference-of-Gaussians preprocessing.
    Mimics retinal processing: enhances edges, removes flat illumination.
    Replaces PatchNormalization in the transform pipeline.
    """
    def __init__(self, sigma1=1.0, sigma2=2.0, resolution=28):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.resolution = resolution

    def __call__(self, pil_image):
        # Resize + grayscale
        img = pil_image.convert('L').resize((self.resolution, self.resolution))
        img_np = np.array(img, dtype=np.float32) / 255.0

        g1 = cv2.GaussianBlur(img_np, (0, 0), self.sigma1)
        g2 = cv2.GaussianBlur(img_np, (0, 0), self.sigma2)
        dog = g1 - g2

        # Normalize to [0, 1]
        dog = dog - dog.min()
        dmax = dog.max()
        if dmax > 0:
            dog = dog / dmax

        return torch.FloatTensor(dog).unsqueeze(0)  # [1, H, W]


class SpatialPyramidDescriptor:
    """
    Multi-level spatial pyramid + HOG descriptor.
    
    Captures WHERE features are in the image (spatial layout)
    and WHAT types of edges dominate (orientation histograms).
    
    This is far more discriminative for indoor scenes than raw pixels
    because rooms differ in spatial layout (bright ceiling vs dark floor,
    edge-rich walls vs smooth floors) rather than pixel-level patterns.
    
    Output is a 1D feature vector normalized to [0, 1].
    """
    def __init__(self, working_res=56, pyramid_levels=(2, 4, 8),
                 hog_cells=4, hog_bins=8):
        self.working_res = working_res
        self.pyramid_levels = pyramid_levels
        self.hog_cells = hog_cells
        self.hog_bins = hog_bins

        # Pre-compute output dimension
        self.n_features = 0
        for level in pyramid_levels:
            self.n_features += level * level * 3  # mean, std, edge_density per cell
        self.n_features += hog_cells * hog_cells * hog_bins  # HOG

    def __call__(self, pil_image):
        # Resize and convert to grayscale numpy
        img = pil_image.convert('L').resize((self.working_res, self.working_res))
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Compute edge magnitude map (Sobel)
        gx = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.arctan2(gy, gx)  # [-pi, pi]
        angle_norm = (angle + np.pi) / (2 * np.pi)  # [0, 1]

        features = []

        # ---- Spatial pyramid: mean, std, edge_density per cell ----
        for level in self.pyramid_levels:
            cell_h = self.working_res // level
            cell_w = self.working_res // level
            for i in range(level):
                for j in range(level):
                    r0, r1 = i * cell_h, (i + 1) * cell_h
                    c0, c1 = j * cell_w, (j + 1) * cell_w
                    cell = img_np[r0:r1, c0:c1]
                    cell_edge = mag[r0:r1, c0:c1]
                    features.append(cell.mean())
                    features.append(cell.std())
                    features.append(cell_edge.mean())

        # ---- HOG: orientation histogram per cell ----
        cell_h = self.working_res // self.hog_cells
        cell_w = self.working_res // self.hog_cells
        for i in range(self.hog_cells):
            for j in range(self.hog_cells):
                r0, r1 = i * cell_h, (i + 1) * cell_h
                c0, c1 = j * cell_w, (j + 1) * cell_w
                cell_mag = mag[r0:r1, c0:c1]
                cell_ang = angle_norm[r0:r1, c0:c1]

                hist = np.zeros(self.hog_bins, dtype=np.float32)
                for b in range(self.hog_bins):
                    lo = b / self.hog_bins
                    hi = (b + 1) / self.hog_bins
                    mask = (cell_ang >= lo) & (cell_ang < hi)
                    hist[b] = cell_mag[mask].sum()

                total = hist.sum()
                if total > 0:
                    hist /= total
                features.extend(hist.tolist())

        features = np.array(features, dtype=np.float32)

        # Per-feature normalization to [0, 1]
        fmin = features.min()
        fmax = features.max()
        if fmax > fmin:
            features = (features - fmin) / (fmax - fmin)

        return torch.FloatTensor(features)  # [D]


def check_dataset_alignment(ds1, ds2):
    """
    Verifies that filenames in two datasets align.
    """
    print("Checking dataset alignment...")
    count = min(len(ds1), len(ds2))
    mismatches = 0
    for i in range(count):
        f1 = os.path.basename(ds1.image_files[i])
        f2 = os.path.basename(ds2.image_files[i])
        if f1 != f2:
            print(f"MISMATCH at index {i}: {f1} vs {f2}")
            mismatches += 1
            if mismatches > 5:
                raise ValueError("Too many mismatches. Datasets are not aligned.")
    
    if mismatches == 0:
        print(f"Verified alignment of {count} images.")
    else:
        raise ValueError(f"Found {mismatches} filename mismatches.")

def get_oxford_10m_subset(sun_csv_path, sun_ins_path):
    """
    Samples frames from the Oxford dataset at ~10-meter intervals 
    using the original timestamps and INS data.
    """
    import csv
    import math
    import numpy as np

    # Load INS data
    ins_timestamps = []
    northing = []
    easting = []
    with open(sun_ins_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            ins_timestamps.append(int(row[0]))
            northing.append(float(row[5]))
            easting.append(float(row[6]))

    ins_timestamps = np.array(ins_timestamps)
    northing = np.array(northing)
    easting = np.array(easting)

    # Read frame timestamps
    frame_ts = []
    frame_names = []
    with open(sun_csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            frame_names.append(row[0])
            orig_name = row[1]
            ts = int(orig_name.replace('.png', ''))
            frame_ts.append(ts)

    sampled_frames = set()
    last_n = None
    last_e = None

    # Interpolate positions
    frame_n = np.interp(frame_ts, ins_timestamps, northing)
    frame_e = np.interp(frame_ts, ins_timestamps, easting)

    for i in range(len(frame_names)):
        n = frame_n[i]
        e = frame_e[i]
        if last_n is None:
            sampled_frames.add(frame_names[i])
            last_n = n
            last_e = e
        else:
            dist = math.sqrt((n - last_n)**2 + (e - last_e)**2)
            if dist >= 10.0:
                sampled_frames.add(frame_names[i])
                last_n = n
                last_e = e

    return sampled_frames

def get_nordland_loaders(train_path, test_path, batch_size=64, max_samples=None, start_idx=0, shuffle_train=True, oxford_10m_sampling=False, resolution=28):
    """
    Creates loaders.
    train_path can be a string or a list of strings (datasets will be concatenated).
    """
    
    # Resize to given resolution (default 28x28)
    # Patch Normalization scales patch size to maintain 16 patches (4x4)
    patch_size = resolution // 4
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)), 
        transforms.Grayscale(),
        transforms.ToTensor(),
        PatchNormalization(patch_size=patch_size)
    ])
    
    valid_files = None
    if oxford_10m_sampling:
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sun_csv_path = os.path.join(ROOT_DIR, 'tools/dataset_imagenames/ORC_Sun_timestamps.csv')
        sun_ins_path = 'ORC/ORC_sun/2015-08-12-15-04-18/ins.csv'
        print(f"Applying Oxford 10-meter spatial sampling using {sun_ins_path}...")
        valid_files = get_oxford_10m_subset(sun_csv_path, sun_ins_path)
        print(f"Sampled {len(valid_files)} frames from Oxford Sun traverse.")

    # 1. Load Test Dataset first (Reference for alignment)
    print(f"Loading Testing Data from: {test_path}")
    test_ds = NordlandDataset(dir_path=test_path, transform=transform, valid_files=valid_files)
    
    # 2. Load Training Dataset(s)
    if isinstance(train_path, str):
        train_paths = [train_path]
    else:
        train_paths = train_path
        
    train_datasets = []
    
    min_len = len(test_ds)
    
    for tp in train_paths:
        print(f"Loading Training Data from: {tp}")
        ds = NordlandDataset(dir_path=tp, transform=transform, valid_files=valid_files)
        
        # Check counts
        if len(ds) != len(test_ds):
            print(f"WARNING: Train ({len(ds)}) and Test ({len(test_ds)}) have different counts!")
            current_min = min(len(ds), len(test_ds))
            if current_min < min_len:
                min_len = current_min
        
        train_datasets.append(ds)

    # 3. Find intersection of filenames to ensure alignment
    common_files = set(test_ds.image_files)
    for ds in train_datasets:
        common_files = common_files.intersection(set(ds.image_files))
    
    common_files = sorted(list(common_files))
    
    if len(common_files) == 0:
        raise ValueError("No common images found across datasets! Cannot align.")
        
    print(f"Found {len(common_files)} aligned images across all datasets.")

    # Apply max_samples and start_idx
    if max_samples is not None:
        print(f"Limiting dataset to {max_samples} samples starting from index {start_idx}.")
        common_files = common_files[start_idx:start_idx+max_samples]
    else:
        common_files = common_files[start_idx:]
        
    min_len = len(common_files)
    
    test_ds.image_files = common_files.copy()
    for ds in train_datasets:
        ds.image_files = common_files.copy()
        
    # 4. Concatenate Train Datasets
    if len(train_datasets) > 1:
        print(f"Concatenating {len(train_datasets)} training datasets...")
        full_train_ds = torch.utils.data.ConcatDataset(train_datasets)
    else:
        full_train_ds = train_datasets[0]
    
    print(f"Creating Loaders (Shuffle Train: {shuffle_train})...")
    train_loader = torch.utils.data.DataLoader(full_train_ds, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Return num_classes based on the logical places (min_len)
    return train_loader, test_loader, min_len

def get_mddrobots_loaders(dataset_root, room_name, batch_size=64, max_samples=None,
                          start_idx=0, shuffle_train=True, frame_skip=4, resolution=28):
    """
    Creates DataLoaders for the MDDRobots indoor VPR dataset.

    MDDRobots has 9 rooms, each with separate train/test traversals.
    Images are sequentially numbered (00000000.png, 00000001.png, ...)
    providing Nordland-style 1:1 correspondence within each room.

    Frame-skip subsampling takes every Nth frame to reduce visual
    overlap between consecutive places (analogous to Oxford 10m sampling).

    Args:
        dataset_root (str): Path to MDDRobots_dataset/ directory.
        room_name (str): Room subdirectory name (e.g., 'Corridor1_RGB').
        batch_size (int): Batch size for DataLoaders.
        max_samples (int or None): Maximum number of places to use.
        start_idx (int): Starting index into the (subsampled) file list.
        shuffle_train (bool): Whether to shuffle the training DataLoader.
        frame_skip (int): Take every Nth frame (1 = no skip, 4 = every 4th).
        resolution (int): Image resolution (default 28).

    Returns:
        train_loader: DataLoader for the reference (train) traversal.
        test_loader: DataLoader for the query (test) traversal.
        num_classes (int): Number of aligned places.
    """
    train_path = os.path.join(dataset_root, "DataSet_GOPRO_RGB_train", room_name)
    test_path = os.path.join(dataset_root, "DataSet_GOPRO_RGB_test1", room_name)

    if not os.path.isdir(train_path):
        raise ValueError(f"MDDRobots train path not found: {train_path}")
    if not os.path.isdir(test_path):
        raise ValueError(f"MDDRobots test path not found: {test_path}")

    # Same preprocessing pipeline as Nordland
    patch_size = resolution // 4
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        PatchNormalization(patch_size=patch_size)
    ])

    print(f"Loading MDDRobots room: {room_name}")
    print(f"  Train path: {train_path}")
    print(f"  Test path:  {test_path}")

    # Load both traversals using the existing NordlandDataset class
    train_ds = NordlandDataset(dir_path=train_path, transform=transform)
    test_ds = NordlandDataset(dir_path=test_path, transform=transform)
    
    # Same subsampling as before
    # ...
    # Wait, instead of replacing everything, I'll just append to the end.


    print(f"  Train images: {len(train_ds)}, Test images: {len(test_ds)}")

    # Find common filenames for alignment (train has 600, test has 500)
    common_files = sorted(
        set(train_ds.image_files).intersection(set(test_ds.image_files))
    )

    if len(common_files) == 0:
        raise ValueError("No common images found between train and test traversals!")

    print(f"  Aligned images (common filenames): {len(common_files)}")

    # Apply frame-skip subsampling: take every Nth frame
    if frame_skip > 1:
        common_files = common_files[::frame_skip]
        print(f"  After frame-skip (every {frame_skip}th): {len(common_files)} places")

    # Apply start_idx and max_samples
    if max_samples is not None:
        print(f"  Limiting to {max_samples} samples starting from index {start_idx}.")
        common_files = common_files[start_idx:start_idx + max_samples]
    else:
        common_files = common_files[start_idx:]

    num_classes = len(common_files)
    print(f"  Final place count: {num_classes}")

    # Update datasets to use only the selected files
    train_ds.image_files = common_files.copy()
    test_ds.image_files = common_files.copy()

    print(f"Creating Loaders (Shuffle Train: {shuffle_train})...")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, num_classes

class RoomClassificationDataset(Dataset):
    """
    Custom Dataset for room-level classification.
    Given a root directory (e.g., DataSet_GOPRO_RGB_train), it finds all room subdirectories.
    Each room gets a unique class label (0 to num_rooms - 1).
    All images within a room are assigned that room's class label.
    """
    def __init__(self, root_dir, transform=None, frame_skip=1, max_samples_per_room=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] # list of (image_path, class_label)
        
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        
        # Get all room directories, sorted so train and test have the same class IDs
        self.rooms = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        for class_id, room_name in enumerate(self.rooms):
            room_path = os.path.join(root_dir, room_name)
            files = sorted([f for f in os.listdir(room_path) if os.path.splitext(f)[1].lower() in valid_exts])
            
            # Apply frame skip
            files = files[::frame_skip]
            
            # Apply max_samples
            if max_samples_per_room is not None:
                files = files[:max_samples_per_room]
                
            for f in files:
                self.samples.append((os.path.join(room_path, f), class_id))
                
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    @property
    def image_files(self):
        # Extract just the basenames to match the behavior of NordlandDataset
        return [os.path.basename(path) for path, _ in self.samples]

    def __getitem__(self, idx):
        img_path, class_label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (28, 28))

        if self.transform:
            image = self.transform(image)
            
        return image, class_label

def get_mddrobots_room_loaders(dataset_root, batch_size=64, max_samples=None,
                               shuffle_train=True, frame_skip=4, resolution=28,
                               descriptor='raw'):
    """
    Creates DataLoaders for MDDRobots room classification.
    
    Args:
        descriptor: 'raw' (default pixel pipeline), 'dog' (DoG preprocessing),
                    'spatial_pyramid' (spatial pyramid + HOG descriptor)
    Returns:
        train_loader, test_loader, num_classes, n_in
    """
    train_root = os.path.join(dataset_root, "DataSet_GOPRO_RGB_train")
    test_root = os.path.join(dataset_root, "DataSet_GOPRO_RGB_test1")

    if not os.path.isdir(train_root) or not os.path.isdir(test_root):
        raise ValueError(f"MDDRobots roots not found: {train_root} or {test_root}")

    if descriptor == 'dog':
        transform = DoGTransform(sigma1=1.0, sigma2=2.0, resolution=resolution)
        n_in = resolution * resolution
        print(f"Using DoG descriptor (N_IN={n_in})")
    elif descriptor == 'spatial_pyramid':
        sp = SpatialPyramidDescriptor(working_res=56)
        transform = sp
        n_in = sp.n_features
        print(f"Using Spatial Pyramid descriptor (N_IN={n_in})")
    else:
        patch_size = resolution // 4
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            PatchNormalization(patch_size=patch_size)
        ])
        n_in = resolution * resolution
        print(f"Using raw pixel descriptor (N_IN={n_in})")

    print("Loading MDDRobots for Room-Level Classification...")
    train_ds = RoomClassificationDataset(train_root, transform=transform, 
                                         frame_skip=frame_skip, max_samples_per_room=max_samples)
    test_ds = RoomClassificationDataset(test_root, transform=transform, 
                                        frame_skip=frame_skip, max_samples_per_room=max_samples)
                                        
    print(f"  Found {len(train_ds.rooms)} rooms: {train_ds.rooms}")
    print(f"  Train images: {len(train_ds)}, Test images: {len(test_ds)}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(train_ds.rooms), n_in
