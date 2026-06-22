"""
networks.py

SNN Architectures.
"""

import torch
import torch.nn as nn
import snntorch as snn
from mechanisms import weight_dependent_stdp, normalize_weights_column, hard_wta_step, homeostatic_threshold_update

class torchVPRSNN(nn.Module):
    """
    Unsupervised Spiking Neural Network for Visual Place Recognition.
    Renamed from DiehlCookVPRSNN.
    
    Architecture:
        Input → Excitatory (E) → Inhibitory (I) → Excitatory
    
    Algorithm Source:
        Diehl, P. U., & Cook, M. (2015). 
        Unsupervised learning of digit recognition using spike-timing-dependent plasticity. 
        Frontiers in computational neuroscience, 9, 99.
    
    Refactoring Logic:
        Separated into modular components (mechanisms, encoders) to mimic 
        standard PyTorch geometric / functional library patterns.
    """

    def __init__(
        self,
        n_in=784,
        n_exc=200,
        t_steps=100,
        beta_e=0.95,
        beta_i=0.90,
        thr_e_init=1.5,
        thr_i=1.0,
        a_plus=5e-4,
        a_minus=5e-6,
        tau_pre=15.0,
        tau_post=15.0,
        w_max=1.0,
        w_ei=1.0,
        w_ie=10.0,
        thr_eta=0.001,
        target_rate=0.01,
        max_samples=100,
        device="cpu",
        wta_mode="hard",
        enable_homeostasis=True,
        enable_weight_norm=True
    ):
        super().__init__()

        self.n_in = n_in
        self.n_exc = n_exc
        self.n_inh = n_exc
        self.t_steps = t_steps
        self.device = device
        self.w_max = w_max
        
        self.wta_mode = wta_mode
        self.enable_homeostasis = enable_homeostasis
        self.enable_weight_norm = enable_weight_norm

        # ----------------------------------------------------
        # Plastic Input → Excitatory weights (STDP)
        # ----------------------------------------------------
        w0 = torch.rand(n_in, n_exc)
        self.w_in_exc = nn.Parameter(w0, requires_grad=False)
        self._normalize_w()

        # ----------------------------------------------------
        # lateral inhibition
        # ----------------------------------------------------
        self.register_buffer("w_exc_inh", torch.eye(n_exc) * w_ei)
        
        w_ie_mat = torch.ones(n_exc, n_exc) * w_ie
        for k in range(n_exc):
            w_ie_mat[k, k] = 0.0
        self.register_buffer("w_inh_exc", w_ie_mat)

        # ----------------------------------------------------
        # Neuron models
        # ----------------------------------------------------
        self.thr_e = nn.Parameter(
            torch.ones(n_exc) * thr_e_init, requires_grad=False
        )

        self.lif_e = snn.Leaky(beta=beta_e, threshold=thr_e_init, init_hidden=False)
        self.lif_i = snn.Leaky(beta=beta_i, threshold=thr_i, init_hidden=False)

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.thr_eta = thr_eta
        self.target_rate = target_rate

    def _normalize_w(self):
        if self.enable_weight_norm:
            normalize_weights_column(self.w_in_exc)

    def forward(self, spk_in, do_stdp=True, monitor=False, state=None, return_state=False):
        """
        Run temporal SNN simulation.
        """
        T, B, _ = spk_in.shape

        if state is None:
            mem_e = torch.zeros(B, self.n_exc, device=self.device)
            mem_i = torch.zeros(B, self.n_inh, device=self.device)
            pre_trace = torch.zeros(B, self.n_in, device=self.device)
            post_trace = torch.zeros(B, self.n_exc, device=self.device)
        else:
            mem_e, mem_i, pre_trace, post_trace = state

        spike_rec = []
        
        history = {
            "pre_spk": [], "post_spk": [],
            "pre_trace": [], "post_trace": [],
            "dw_plus": 0, "dw_minus": 0
        } if monitor else None

        for t in range(T):
            pre_spk = spk_in[t]

            # Input → Excitatory
            cur_e = pre_spk @ self.w_in_exc
            self.lif_e.threshold = self.thr_e
            spk_e, mem_e = self.lif_e(cur_e, mem_e)

            # Winner-Take-All
            if self.wta_mode == "hard":
                spk_e = hard_wta_step(spk_e, cur_e)

            # Exc → Inh → Exc inhibition
            if self.wta_mode in ["hard", "soft"]:
                cur_i = spk_e @ self.w_exc_inh
                spk_i, mem_i = self.lif_i(cur_i, mem_i)
                inh_e = spk_i @ self.w_inh_exc

                mem_e -= inh_e
                mem_e = torch.clamp(mem_e, min=-2.0)

            spike_rec.append(spk_e)

            # STDP traces
            pre_trace = pre_trace * (1 - 1/self.tau_pre) + pre_spk
            post_trace = post_trace * (1 - 1/self.tau_post) + spk_e

            if monitor:
                history["pre_spk"].append(pre_spk[0].cpu().numpy())
                history["post_spk"].append(spk_e[0].cpu().numpy())
                history["pre_trace"].append(pre_trace[0].cpu().numpy())
                history["post_trace"].append(post_trace[0].cpu().numpy())

            if do_stdp:
                dw_p, dw_m = weight_dependent_stdp(
                    pre_spk, spk_e, pre_trace, post_trace, 
                    self.w_in_exc, 
                    self.a_plus, self.a_minus, self.w_max
                )
                self._normalize_w()
                
                if monitor:
                    history["dw_plus"] += dw_p.cpu().numpy()
                    history["dw_minus"] += dw_m.cpu().numpy()

            # Homeostatic threshold adaptation
            if do_stdp and self.enable_homeostasis: # Only adapt during training
                homeostatic_threshold_update(self.thr_e, spk_e, self.target_rate, self.thr_eta)
            
        new_state = (mem_e, mem_i, pre_trace, post_trace)
        
        if return_state:
            if monitor:
                return torch.stack(spike_rec, dim=0), history, new_state
            return torch.stack(spike_rec, dim=0), new_state
        else:
            if monitor:
                return torch.stack(spike_rec, dim=0), history
            return torch.stack(spike_rec, dim=0)
