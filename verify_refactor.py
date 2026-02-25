
import torch
import sys
import os

print("Testing imports...")
try:
    from encoders import RateEncoder
    from mechanisms import weight_dependent_stdp, hard_wta_step
    from networks import torchVPRSNN
    from utils import Logger
    print("Imports successful.")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_mechanisms():
    print("Testing mechanisms...")
    B, N = 2, 5
    spikes = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    currents = torch.tensor([[0.5, 0.1, 0.8, 0.2, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1]])
    
    # Test Hard WTA
    out = hard_wta_step(spikes.clone(), currents)
    print("Hard WTA Input Spikes:", spikes[0])
    print("Hard WTA Input Currents:", currents[0])
    print("Hard WTA Output:", out[0])
    
    assert out[0, 2] == 1.0, "WTA failed to select max current"
    assert out[0, 0] == 0.0, "WTA failed to suppress non-max"
    
    # Test STDP signature (passed dummy values)
    w = torch.rand(3, 3)
    pre = torch.rand(1, 3)
    post = torch.rand(1, 3)
    pre_tr = torch.rand(1, 3)
    post_tr = torch.rand(1, 3)
    
    dw_p, dw_m = weight_dependent_stdp(pre, post, pre_tr, post_tr, w, 0.01, 0.01, 1.0)
    print("STDP executed successfully.")

def test_network():
    print("Testing torchVPRSNN instantiation...")
    net = torchVPRSNN(n_in=10, n_exc=5, t_steps=10, device='cpu')
    x = torch.rand(2, 1, 1, 10) # B, C, H, W (1D img)
    encoder = RateEncoder(t_steps=10)
    spk = encoder(x)
    print("Encoder output shape:", spk.shape)
    
    out = net(spk, do_stdp=False)
    print("Network output shape:", out.shape)

if __name__ == "__main__":
    test_mechanisms()
    test_network()
    print("Verification passed!")
