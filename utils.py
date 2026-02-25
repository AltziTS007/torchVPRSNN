"""
utils.py

Shared utilities for SNN experiments, including logging and monitoring.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

class Logger(object):
    """
    Simple logger that duplicates stdout to a file.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def plot_stdp_monitor(history, epoch, save_dir):
    """
    Visualizes STDP dynamics for a single sample over time.
    
    Args:
        history (dict): Dictionary with keys 'pre_spk', 'post_spk', etc.
        epoch (int): Current epoch number.
        save_dir (str): Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    pre_spk = np.array(history["pre_spk"]) # [T, n_in]
    post_spk = np.array(history["post_spk"]) # [T, n_exc]
    pre_trace = np.array(history["pre_trace"]) # [T, n_in]
    post_trace = np.array(history["post_trace"]) # [T, n_exc]
    dw_plus = history["dw_plus"] # [n_in, n_exc]
    dw_minus = history["dw_minus"] # [n_in, n_exc]
    
    T = pre_spk.shape[0]
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Spikes (Raster)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(pre_spk.T, aspect='auto', cmap='binary', interpolation='nearest', origin='lower')
    ax1.set_title("Input Spikes (Pre) [Sample 0]")
    ax1.set_ylabel("Neuron ID")
    ax1.set_xlabel("Time Step")
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(post_spk.T, aspect='auto', cmap='binary', interpolation='nearest', origin='lower')
    ax2.set_title("Output Spikes (Post) [Sample 0]")
    ax2.set_xlabel("Time Step")
    
    # 2. Traces (Average or Sample)
    ax3 = fig.add_subplot(gs[1, 0])
    # Plot average trace behavior
    ax3.plot(pre_trace.mean(axis=1), label="Mean Pre Trace", color='blue', linewidth=2)
    # Plot a few active neurons (if any fired)
    active_pre = np.where(pre_spk.sum(axis=0) > 0)[0]
    if len(active_pre) > 0:
        for idx in active_pre[:3]:
            ax3.plot(pre_trace[:, idx], alpha=0.5, linestyle=':', label=f"Pre Neuron {idx}")
    ax3.set_title("Pre Trace Evolution")
    ax3.legend()
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(post_trace.mean(axis=1), label="Mean Post Trace", color='red', linewidth=2)
    active_post = np.where(post_spk.sum(axis=0) > 0)[0]
    if len(active_post) > 0:
        for idx in active_post[:3]:
            ax4.plot(post_trace[:, idx], alpha=0.5, linestyle=':', label=f"Post Neuron {idx}")
    ax4.set_title("Post Trace Evolution")
    ax4.legend()
    
    # 3. Accumulated dW
    ax5 = fig.add_subplot(gs[2, 0])
    # dw_plus is [n_in, n_exc]. Transpose for visualization if needed.
    im5 = ax5.imshow(dw_plus.T, aspect='auto', cmap='hot', interpolation='nearest', origin='lower')
    ax5.set_title("Accumulated dW Plus (LTP)")
    ax5.set_ylabel("Post ID")
    ax5.set_xlabel("Pre ID")
    plt.colorbar(im5, ax=ax5)
    
    ax6 = fig.add_subplot(gs[2, 1])
    im6 = ax6.imshow(dw_minus.T, aspect='auto', cmap='cool', interpolation='nearest', origin='lower')
    ax6.set_title("Accumulated dW Minus (LTD)")
    ax6.set_xlabel("Pre ID")
    plt.colorbar(im6, ax=ax6)
    
    plt.tight_layout()
    plt.suptitle(f"STDP Dynamics - Epoch {epoch}", y=1.02)
    plt.savefig(os.path.join(save_dir, f"stdp_epoch_{epoch}.png"))
    plt.close()
