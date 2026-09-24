"""
neuronal_assignments.py

Implements Section B (Weighted Assignments) and Section C (Probability-based Assignments)
from Sliding Window et al. "Spiking Neural Networks for Visual Place Recognition via Weighted Neuronal Assignments".
"""

import torch
from tqdm import tqdm
import time
import numpy as np
import math

def get_training_spike_counts(net, encoder, loader, n_exc, n_classes, device="cpu"):
    """
    Computes the full Spike Count Matrix S^R for the training set.
    
    Args:
        net: Trained SNN
        encoder: Spike encoder
        loader: Training DataLoader
        n_exc: Number of excitatory neurons
        n_classes: Number of place classes
        device: 'cpu' or 'cuda'
        
    Returns:
        S_R (Tensor): [n_exc, n_classes]
            S_R[i, l] = Total spikes of neuron i for place l during training.
    """
    net.eval()
    
    # S_R: [n_exc, n_classes]
    S_R = torch.zeros(n_exc, n_classes, device=device)
    
    print("Computing Training Spike Counts (S^R)...")
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Computing S^R"):
            xb = xb.to(device)
            yb = yb.to(device)
            
            spk_in = encoder(xb)
            
            # Forward pass (no plasticity)
            # exc_counts: [batch_size, n_exc]
            exc_counts = net(spk_in, do_stdp=False).sum(dim=0)
            
            # Accumulate
            for i in range(len(yb)):
                label = yb[i].item()
                if label < n_classes:
                    S_R[:, label] += exc_counts[i]
                    
    return S_R

def weighted_assignment_inference(S_Q, S_R, gamma=0.02):
    """
    Implements Weighted Neuronal Assignments (Section B).
    
    Args:
        S_Q (Tensor): [n_query, n_exc] - Spike counts for query images
        S_R (Tensor): [n_exc, n_classes] - Training spike counts (from get_training_spike_counts)
        gamma (float): Regularization parameter (fraction of R). Default 0.02 (2%).
        
    Returns:
        scores (Tensor): [n_query, n_classes] - Similarity scores
    """
    n_query, n_exc = S_Q.shape
    n_exc_r, n_classes = S_R.shape
    assert n_exc == n_exc_r, "Mismatch in neurons between Query and Train"
    
    device = S_Q.device
    
    # Ensure S_R is on the same device
    S_R = S_R.to(device)
    
    # --------------------------------------------------------
    # Step 1: Regularization by Involvement (Eq 6)
    # --------------------------------------------------------
    # omega_i: Number of places learned by neuron i.
    # We define "learned" as having non-zero response in S_R.
    # omega: [n_exc]
    omega = (S_R > 0).sum(dim=1).float()
    
    threshold = gamma * n_classes
    
    # If omega_i <= gamma * R: S' = S
    # If omega_i > gamma * R:  S' = S * (1 / omega_i)
    
    # Create regularization factor vector [n_exc]
    # Default is 1.0
    reg_factor = torch.ones(n_exc, device=device)
    
    # Find neurons that need regularization
    mask = omega > threshold
    
    # Avoid division by zero if omega is 0 (though mask handles > threshold)
    safe_omega = omega.clone()
    safe_omega[safe_omega == 0] = 1.0
    
    reg_factor[mask] = 1.0 / safe_omega[mask]
    
    # Apply to S_R rows
    # S_prime: [n_exc, n_classes]
    S_prime = S_R * reg_factor.unsqueeze(1)
    
    # --------------------------------------------------------
    # Step 2: Normalization by Response Strength (Eq 7)
    # --------------------------------------------------------
    # "Weight by ratio S^R_{i,l} / sum_m S^R_{i,m}"
    # Denominator: Total spikes of neuron i across ALL places
    # neuron_total_spikes: [n_exc]
    neuron_total_spikes = S_R.sum(dim=1)
    
    # Avoid div by zero
    neuron_total_spikes[neuron_total_spikes == 0] = 1.0
    
    strength_factor = S_R / neuron_total_spikes.unsqueeze(1) # [n_exc, n_classes]
    
    # S_double_prime = S_prime * strength_factor
    S_double_prime = S_prime * strength_factor
    
    # --------------------------------------------------------
    # Inference / Scoring so far
    # --------------------------------------------------------
    # the score for place l would be: sum_i (S^Q_i * S''_i,l)
    
    # scores_intermediate: [n_query, n_classes]
    # S_Q: [n_query, n_exc]
    # S_double_prime: [n_exc, n_classes]
    scores_intermediate = torch.matmul(S_Q, S_double_prime)
    
    # --------------------------------------------------------
    # Step 3: Penalize relevant neurons that did not spike (Eq 8)
    # --------------------------------------------------------
    # "Down-weights place labels l when not all neurons m that have learned that place label, fired"
    # Factor = (Sum of training spikes for place l of CURRENTLY ACTIVE neurons) 
    #          / (Total training spikes for place l)
    
    # Denominator: Total S^R spikes for place l (across all neurons)
    # place_total_spikes: [n_classes]
    place_total_spikes = S_R.sum(dim=0)
    place_total_spikes[place_total_spikes == 0] = 1.0 # Avoid div/0
    
    # Numerator calculation is query-dependent.
    # For each query, we identify active neurons (S^Q > 0).
    # Then sum S^R_{m,l} for those active m.
    
    # Indicator of active neurons in query: I_Q [n_query, n_exc]
    I_Q = (S_Q > 0).float()
    
    # Numerator: Matmul(I_Q, S_R) -> [n_query, n_classes]
    # This sums S_R columns, but only for rows (neurons) active in Q.
    numerator = torch.matmul(I_Q, S_R)
    
    # Factor matrix: [n_query, n_classes]
    penalty_factor = numerator / place_total_spikes.unsqueeze(0)
    
    # Final Scores
    scores_final = scores_intermediate * penalty_factor
    
    return scores_final

def probability_based_assignment(scores):
    """
    Implements Probability-Based Neuronal Assignment (Section C).
    Applies Min-Max normalization per query to convert scores to probabilities.
    
    Args:
        scores (Tensor): [n_query, n_classes] - Raw similarity scores
        
    Returns:
        prob_scores (Tensor): [n_query, n_classes] - Normalized scores
    """
    # Min and Max per query (row)
    # min_val: [n_query, 1]
    min_val, _ = scores.min(dim=1, keepdim=True)
    max_val, _ = scores.max(dim=1, keepdim=True)
    
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0 # Avoid div/0
    
    # Min-Max Normalization
    scores_norm = (scores - min_val) / range_val
    
    # Scale by total sum (make it a PDF)
    row_sums = scores_norm.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    
    prob_scores = scores_norm / row_sums
    
    return prob_scores

def sliding_window_aggregation(prob_scores, targets, k, rule="product", velocity_factor=1.0):
    """
    Implements a sliding window aggregation over sequential queries.
    
    Args:
        prob_scores (Tensor): [n_query, n_classes] - Probability scores per frame
        targets (Tensor or list): [n_query] - Ground truth targets
        k (int): Window size
        rule (str): 'product' or 'mean'
        velocity_factor (float): Simulates traverse velocity mismatch. 1.0 means exactly 1 frame per place.
        
    Returns:
        agg_scores (Tensor): [n_valid_queries, n_classes]
        valid_targets (list): [n_valid_queries]
        latency_ms (float): Average per-query latency overhead in milliseconds
    """
    n_query, n_classes = prob_scores.shape
    device = prob_scores.device
    
    if not isinstance(targets, torch.Tensor):
        targets_t = torch.tensor(targets, device=device)
    else:
        targets_t = targets.to(device)
        
    # 1. Find discontinuities (resets)
    resets = []
    for i in range(1, n_query):
        if targets_t[i] != targets_t[i-1] + 1:
            resets.append(i)
            
    # 2. Determine valid windows (discard k-1 frames at the start and after any reset)
    valid_indices = []
    for i in range(n_query):
        # Find the start of the current sequence (latest reset <= i)
        seq_start = 0
        for r in resets:
            if r <= i:
                seq_start = r
        
        # We need at least k frames in the current sequence
        if i - seq_start >= k - 1:
            valid_indices.append(i)
            
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        return torch.empty((0, n_classes), device=device), [], 0.0
        
    # We will measure latency for the aggregation operations only
    agg_scores = torch.zeros((n_valid, n_classes), device=device)
    valid_targets = []
    
    # Force sync for precise timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    eps = 1e-8
    # NOTE: Compute log on-the-fly per window to avoid OOM on large datasets.
    # Pre-computing the full log_probs tensor would duplicate a [n_query, n_classes]
    # matrix in memory, which exceeds GPU memory for datasets > ~10k samples.
    
    for valid_idx, i in enumerate(valid_indices):
        valid_targets.append(targets[i])
        
        # Window bounds: [i - k + 1, i]
        start_idx = i - k + 1
        end_idx = i + 1
        
        if rule == "product":
            window_log_probs = torch.log(prob_scores[start_idx:end_idx] + eps)
            shifted_window = torch.full_like(window_log_probs, fill_value=np.log(eps))
            
            for m in range(k):
                raw_shift = (k - 1 - m) * velocity_factor
                # Use arithmetic rounding away from zero
                shift = math.floor(raw_shift + 0.5) if raw_shift >= 0 else math.ceil(raw_shift - 0.5)
                
                # Out of bounds handling
                if abs(shift) >= n_classes:
                    continue # shifted_window[m] remains fill_value (log(eps) or 0)
                
                if shift == 0:
                    shifted_window[m] = window_log_probs[m]
                elif shift > 0:
                    shifted_window[m, shift:] = window_log_probs[m, :-shift]
                else: # shift < 0
                    abs_shift = -shift
                    shifted_window[m, :-abs_shift] = window_log_probs[m, abs_shift:]
                    
            sum_log_probs = shifted_window.sum(dim=0)
            
            # Convert back to probabilities and normalize
            # Subtract max for numerical stability before exp
            max_log_prob = sum_log_probs.max()
            unnorm_probs = torch.exp(sum_log_probs - max_log_prob)
            
            agg = unnorm_probs / unnorm_probs.sum()
            agg_scores[valid_idx] = agg
        else: # mean
            window_probs = prob_scores[start_idx:end_idx]
            shifted_window = torch.zeros_like(window_probs)
            
            for m in range(k):
                raw_shift = (k - 1 - m) * velocity_factor
                # Use arithmetic rounding away from zero
                shift = math.floor(raw_shift + 0.5) if raw_shift >= 0 else math.ceil(raw_shift - 0.5)
                
                # Out of bounds handling
                if abs(shift) >= n_classes:
                    continue # shifted_window[m] remains 0
                
                if shift == 0:
                    shifted_window[m] = window_probs[m]
                elif shift > 0:
                    shifted_window[m, shift:] = window_probs[m, :-shift]
                else: # shift < 0
                    abs_shift = -shift
                    shifted_window[m, :-abs_shift] = window_probs[m, abs_shift:]
                    
            agg = shifted_window.mean(dim=0)
            # Re-normalize just in case shift caused some mass to fall off the edge
            agg_sum = agg.sum()
            if agg_sum > 0:
                agg = agg / agg_sum
            agg_scores[valid_idx] = agg
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    latency_ms = ((end_time - start_time) / n_valid) * 1000.0
    
    return agg_scores, valid_targets, latency_ms

