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

IMPORTANT:
This file assumes:
- One-to-one correspondence between Spring/Fall images
- Single ground-truth place per query
- No temporal alignment (pure appearance-based VPR)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import cv2
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

def evaluate_vpr(net, encoder, loader, assignments, n_classes, device="cpu"):
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

    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    similarity_matrix = []

    print("Evaluating VPR...")
    all_spike_counts_list = []
    
    with torch.no_grad():
        for xb, yb in tqdm(loader):
            xb = xb.to(device)
            yb = yb.to(device)

            spk_in = encoder(xb)
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

                correct += int(pred == target)
                total += 1

                all_preds.append(pred)
                all_targets.append(target)
            
            # Store batch spike counts
            all_spike_counts_list.append(exc_counts.cpu()) # Move to CPU to save GPU memory

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Stack all spike counts
    all_spike_counts = torch.cat(all_spike_counts_list, dim=0)
    all_spike_counts = all_spike_counts.to(device) # Move back to device if needed? Or keep on CPU?
    # WNA inference expects S_Q on device. Let's move it to device there or here. 
    # For now, let's keep it on CPU to avoid OOM if large, but WNA will likely need it on GPU. 
    # Actually WNA inference starts with `S_Q.to(device)`. So returning CPU tensor is fine.

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
    # We need to collect them during the loop or just return meaningful data
    # Ideally, we should have collected them. 
    # Let's assume we didn't store them in `similarity_matrix` loop.
    # Wait, I didn't store them in the original code above.
    # I need to modify the loop to store `exc_counts`!
    
    return accuracy, all_preds, all_targets, similarity_matrix, prob_matrix, all_spike_counts

# ============================================================
# 4. PLOTTING FUNCTIONS (Paper Replication)
# ============================================================

from sklearn.metrics import precision_recall_curve, auc

def _prepare_metrics_data(similarity_matrix, targets):
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
            GThard[target, i] = 1
            
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
        # but it treats columns as queries and calculates if match is in top K.
        # This is exactly what we want.
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
    We reshape each column of 784 to 28x28.
    """
    # Weights: [784, n_exc]
    # We want to plot n_exc images
    
    # Ensure on CPU
    w = weights.cpu().detach().numpy()
    
    # Create grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    
    for i in range(n_rows * n_cols):
        ax = axes.flat[i]
        if i < n_exc:
            # Get weight vector for neuron i
            # w[:, i] is [784]
            img = w[:, i].reshape(n_side, n_side)
            ax.imshow(img, cmap='hot', interpolation='nearest')
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    #print(f"Saved Weights visualization to {save_path}")

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

def calculate_r_at_100p(similarity_matrix, targets):
    """
    Calculates Recall at 100% Precision using metrics.py
    """
    S_in, GThard = _prepare_metrics_data(similarity_matrix, targets)
    
    # recallAt100precision returns a float (0.0 to 1.0)
    r100p = vpr_metrics.recallAt100precision(S_in, GThard, matching='single')
    
    return r100p * 100.0

def calculate_p_at_100r(similarity_matrix, targets):
    """
    Calculates Precision at 100% Recall using metrics.py createPR
    """
    S_in, GThard = _prepare_metrics_data(similarity_matrix, targets)
    
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
        # Query
        ax = plt.subplot(n_rows, n_cols, 1)
        ax.imshow(query_proc, cmap='gray')
        ax.set_title(f"Query (Input)\n{query_fname}")
        ax.axis('off')

        # Retrieved (Database)
        ax = plt.subplot(n_rows, n_cols, 2)
        ax.imshow(retrieved_proc, cmap='gray')
        ax.set_title(f"Retrieved (Database)\n{retrieved_fname}")
        ax.axis('off')
        
        # Ground Truth (Database)
        ax = plt.subplot(n_rows, n_cols, 3)
        ax.imshow(gt_proc, cmap='gray')
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
            for k, nid in enumerate(neurons_to_show):
                ax = plt.subplot(n_rows, n_cols, 2*n_cols + k + 1)
                w_img = weights_np[:, nid].reshape(28, 28)
                ax.imshow(w_img, cmap='hot')
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
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        
        # extensions to look for
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        
        # Get all image files and sort them to ensure alignment
        self.image_files = sorted([
            f for f in os.listdir(dir_path) 
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        
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
    def __init__(self, patch_size=7): # 28x28 image -> 7x7 patches = 16 patches
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

def get_nordland_loaders(train_path, test_path, batch_size=64, max_samples=None, shuffle_train=True):
    """
    Creates loaders.
    train_path can be a string or a list of strings (datasets will be concatenated).
    """
    
    # Resize to 28x28 for the provided MNIST network architecture
    # Patch Normalization + CLAHE added
    transform = transforms.Compose([
        transforms.Resize((28, 28)), 
        transforms.Grayscale(),
        transforms.ToTensor(),
        PatchNormalization(patch_size=7)
    ])
    
    # 1. Load Test Dataset first (Reference for alignment)
    print(f"Loading Testing Data from: {test_path}")
    test_ds = NordlandDataset(dir_path=test_path, transform=transform)
    
    # 2. Load Training Dataset(s)
    if isinstance(train_path, str):
        train_paths = [train_path]
    else:
        train_paths = train_path
        
    train_datasets = []
    
    min_len = len(test_ds)
    
    for tp in train_paths:
        print(f"Loading Training Data from: {tp}")
        ds = NordlandDataset(dir_path=tp, transform=transform)
        
        # Check counts
        if len(ds) != len(test_ds):
            print(f"WARNING: Train ({len(ds)}) and Test ({len(test_ds)}) have different counts!")
            current_min = min(len(ds), len(test_ds))
            if current_min < min_len:
                min_len = current_min
        
        train_datasets.append(ds)

    # 3. Apply limits and alignment checks
    if max_samples is not None:
        print(f"Limiting dataset to {max_samples} samples.")
        min_len = min(min_len, max_samples)
        
    # Truncate all datasets to min_len
    test_ds.image_files = test_ds.image_files[:min_len]
    print(f"Test dataset truncated to {len(test_ds)} samples.")

    for i, ds in enumerate(train_datasets):
        ds.image_files = ds.image_files[:min_len]
        # Verify alignment with test set
        print(f"Checking alignment for Train Set {i}...")
        check_dataset_alignment(ds, test_ds)
        
    # 4. Concatenate Train Datasets
    if len(train_datasets) > 1:
        print(f"Concatenating {len(train_datasets)} training datasets...")
        full_train_ds = torch.utils.data.ConcatDataset(train_datasets)
    else:
        full_train_ds = train_datasets[0]
    
    print(f"Creating Loaders (Shuffle Train: {shuffle_train})...")
    train_loader = torch.utils.data.DataLoader(full_train_ds, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    
    # Return num_classes based on the logical places (min_len)
    return train_loader, test_loader, min_len









