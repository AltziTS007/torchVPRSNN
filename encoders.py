"""
encoders.py

Encoders for converting static data (images) into spike trains.
Standard PyTorch module structure.
"""

import torch
import torch.nn as nn
from snntorch import spikegen

class RateEncoder(nn.Module):
    """
    Rate-based Poisson spike encoder.

    Converts static image intensities into spike trains using
    Poisson firing with rates proportional to pixel intensity.

    This is biologically weak but computationally stable and
    commonly used in SNN baselines.

    Input:
        x: Tensor [B, C, H, W]
    Output:
        spk: Tensor [T, B, N]
            T = number of timesteps
            N = flattened input dimension (H*W*C)

    Notes:
    - No gradients are propagated (torch.no_grad)
    - rate_scale critically affects firing sparsity
    """

    def __init__(self, t_steps=100, rate_scale=0.2):
        """
        Args:
            t_steps (int): Number of simulation timesteps
            rate_scale (float): Scaling factor for firing probability
        """
        super().__init__()
        self.t_steps = t_steps
        self.rate_scale = rate_scale

    @torch.no_grad()
    def forward(self, x):
        """
        Generate Poisson spike trains from input tensor.

        Args:
            x (Tensor): [B, C, H, W] or [B, 1, 28, 28]

        Returns:
            spk (Tensor): [T, B, N]
        """
        B = x.size(0)

        # Flatten spatial dimensions → feature vector
        rates = x.view(B, -1) * self.rate_scale

        # Enforce valid firing probabilities
        rates = rates.clamp(0.0, 1.0)

        # Generate Poisson spike trains
        spk = spikegen.rate(rates, num_steps=self.t_steps)

        # Optional telemetry (rarely printed)
        if torch.rand(1).item() < 0.01:
            print(f"DEBUG: Input Spike Rate: {spk.float().mean().item():.4f}")

        return spk
