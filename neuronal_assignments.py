"""
neuronal_assignments.py

Implements Section B (Weighted Assignments) and Section C (Probability-based Assignments)
from Hussaini et al. "Spiking Neural Networks for Visual Place Recognition via Weighted Neuronal Assignments".
"""

import torch
from tqdm import tqdm

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
    # At this point, for a standard weighted voting (without Step 3), 
    # the score for place l would be: sum_i (S^Q_i * S''_i,l)
    # But Step 3 is a "normalization step" on the FINAL score for place l.
    
    # Let's compute the intermediate scores first.
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
    
    # We can do this with matrix multiplication.
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
