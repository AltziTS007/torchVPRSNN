import os
import argparse
import torch
from PIL import Image
from torchvision import transforms

from snn_model import set_seed
from encoders import RateEncoder
from networks import torchVPRSNN
from vprsnn_evaluation import PatchNormalization

def main():
    parser = argparse.ArgumentParser(description="Predict Room using trained SNN model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    set_seed(42)

    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        return
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found.")
        return

    # Load checkpoint
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)
    
    if "state_dict" not in checkpoint or "assignments" not in checkpoint:
        print("Error: The model checkpoint is missing the 'assignments' or 'state_dict'.")
        print("Please retrain the model and save it using --save-model.")
        return

    resolution = checkpoint.get("resolution", 56)
    n_exc = checkpoint.get("n_exc", 800)
    assignments = checkpoint["assignments"].to(device)
    num_classes = checkpoint.get("num_classes", assignments.max().item() + 1)
    room_names = checkpoint.get("room_names", [f"Room_{i}" for i in range(num_classes)])

    # Initialize network
    n_in = resolution * resolution
    net = torchVPRSNN(
        n_in=n_in,
        n_exc=n_exc,
        device=device
    ).to(device)
    
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    
    encoder = RateEncoder(
        time_steps=250,
        dt=1.0,
        device=device
    )

    # Preprocess image
    patch_size = resolution // 4
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        PatchNormalization(patch_size=patch_size)
    ])
    
    print(f"Processing image {args.image}...")
    img = Image.open(args.image).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # Add batch dim

    # Encode to spikes
    x_flat = x.view(x.size(0), -1)
    spk_in = encoder(x_flat)  # [time_steps, 1, n_in]

    # Run inference
    with torch.no_grad():
        spk_rec, _ = net(spk_in, do_stdp=False)
        
    # spk_rec is [time_steps, 1, n_exc]
    spike_counts = spk_rec.sum(dim=0)  # [1, n_exc]
    
    # Calculate similarity scores (Standard voting)
    # Each neuron casts votes for its assigned class, weighted by its spike count
    scores = torch.zeros((1, num_classes), device=device)
    for c in range(num_classes):
        # find neurons assigned to class c
        neurons_c = (assignments == c).nonzero(as_tuple=True)[0]
        if len(neurons_c) > 0:
            scores[0, c] = spike_counts[0, neurons_c].sum()
            
    # Normalize by number of neurons assigned to each class
    for c in range(num_classes):
        num_assigned = (assignments == c).sum().float()
        if num_assigned > 0:
            scores[0, c] /= num_assigned

    pred_class = scores.argmax(dim=1).item()
    pred_room = room_names[pred_class]
    
    print("\n--- Prediction Results ---")
    print(f"Predicted Class ID: {pred_class}")
    print(f"Predicted Room:     {pred_room}")
    print("-" * 26)
    
    print("\nConfidence Scores across all rooms:")
    for c in range(num_classes):
        print(f"  {room_names[c]}: {scores[0, c].item():.2f}")

if __name__ == "__main__":
    main()
