"""
mechanisms.py

Functional implementations of SNN mechanisms:
- STDP (Spike-Timing Dependent Plasticity)
- WTA (Winner-Take-All) inhibition
- Threshold adaptation

Inspired by Diehl & Cook (2015) and standard neuromorphic computing practices.
"""

import torch

def normalize_weights_column(weights):
    """
    Column-wise L2 normalization of synaptic weights.
    Prevents runaway excitation.
    
    Args:
        weights (Tensor): [n_in, n_out]
    """
    with torch.no_grad():
        col_norm = torch.sqrt((weights ** 2).sum(dim=0, keepdim=True)) + 1e-6
        weights.div_(col_norm)

def weight_dependent_stdp(
    pre_spk, 
    post_spk, 
    pre_trace, 
    post_trace, 
    weights, 
    a_plus, 
    a_minus, 
    w_max
):
    """
    Performs one step of Weight-Dependent STDP.
    
    Update rule:
        Δw = (η_plus * (w_max - w) * pre_trace * post_spk) 
           - (η_minus * w * post_trace * pre_spk)
           
    Args:
        pre_spk (Tensor): [B, n_in]
        post_spk (Tensor): [B, n_out]
        pre_trace (Tensor): [B, n_in]
        post_trace (Tensor): [B, n_out]
        weights (Tensor): [n_in, n_out] (updated in-place)
        a_plus (float): Learning rate for potentiation
        a_minus (float): Learning rate for depression
        w_max (float): Maximum weight bound
        
    Returns:
        dw_plus, dw_minus (Tensor): The computed updates for monitoring.
    """
    # Compute outer products for batch
    # dw_plus[i,j] = sum_over_batch(pre_trace[b,i] * post_spk[b,j])
    dw_plus = torch.einsum("bi,bj->ij", pre_trace, post_spk)
    dw_minus = torch.einsum("bj,bi->ij", post_trace, pre_spk)

    ltp_scale = (w_max - weights)
    ltd_scale = weights

    dw = a_plus * ltp_scale * dw_plus \
       - a_minus * ltd_scale * dw_minus

    weights += dw
    weights.clamp_(0.0, w_max)
    
    # Normally one would normalize after this, but we leave it to the caller
    # to allow flexibility.
    
    return dw_plus, dw_minus

def hard_wta_step(spikes, currents):
    """
    Applies Hard Winner-Take-All (WTA) to a batch of spikes.
    Only the neuron with the highest input current is allowed to spike.
    
    Args:
        spikes (Tensor): [B, N] - The candidates (already thresholded)
        currents (Tensor): [B, N] - The membrane potentials or currents
        
    Returns:
        spikes (Tensor): [B, N] - Filtered spikes (at most 1 per batch item)
    """
    B = spikes.size(0)
    
    # If any neuron spiked in the batch item
    # (We operate loosely here: if multiple spiked, we look at currents)
    
    # We want to perform this per-sample.
    # Logic: if spikes.sum() > 0, find argmax(current) and suppress others.
    
    # Check which samples have spikes
    has_spikes = spikes.sum(dim=1) > 0 # [B]
    
    if has_spikes.any():
        # Identify winner for each sample based on current
        winner_idx = currents.argmax(dim=1) # [B]
        
        # Reset all spikes
        spikes.zero_()
        
        # Restore spike only for the winner, IF it originally spiked?
        # Diehl & Cook typically forces a spike if potential > thresh, 
        # but if multiple cross thresh, only max potential spikes.
        # So yes, we simply set the winner to 1. 
        # CAUTION: If the "winner" didn't actually cross threshold (current < thresh),
        # this logic might force a spike where there wasn't one if we aren't careful.
        # But usually this is called AFTER `lif` returns spikes.
        # If `lif` returns NO spikes, `has_spikes` is False, so we do nothing.
        # If `lif` returns spikes, we pick the best one.
        
        # Only set spike for winners where `has_spikes` is True
        batch_indices = torch.nonzero(has_spikes).squeeze()
        if batch_indices.dim() == 0: # single item case handling
             batch_indices = batch_indices.unsqueeze(0)
             
        # Select winners for those indices
        winners = winner_idx[batch_indices]
        
        spikes[batch_indices, winners] = 1.0
        
    return spikes

def homeostatic_threshold_update(thresholds, spikes, target_rate, eta, min_thresh=0.6, max_thresh=100.0):
    """
    Updates firing thresholds to maintain a target firing rate.
    
    Args:
        thresholds (Tensor): [N] (updated in-place)
        spikes (Tensor): [B, N]
        target_rate (float): Target probability
        eta (float): Learning rate
    """
    mean_rate = spikes.mean(dim=0) # Average over batch
    thresholds += eta * (mean_rate - target_rate)
    thresholds.clamp_(min_thresh, max_thresh)
