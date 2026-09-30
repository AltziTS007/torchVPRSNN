# API Reference — VPR-SNN

This document summarizes the main modules, classes, and functions in the repository.

- `encoders.py`
  - `RateEncoder(t_steps=100, rate_scale=0.2)` : torch.nn.Module
    - Forward: `forward(x)` → returns spikes as Tensor `[T, B, N]` (Poisson rate encoding).

- `mechanisms.py`
  - `normalize_weights_column(weights)` : in-place column-wise L2 normalization for `[n_in, n_out]` weight matrices.
  - `weight_dependent_stdp(pre_spk, post_spk, pre_trace, post_trace, weights, a_plus, a_minus, w_max)` : computes LTP/LTD updates (returns `dw_plus, dw_minus`) and updates `weights` in-place.
  - `hard_wta_step(spikes, currents)` : enforces a hard per-sample Winner-Take-All on spike candidates.
  - `homeostatic_threshold_update(thresholds, spikes, target_rate, eta, min_thresh=0.6, max_thresh=3.0)` : adjusts firing thresholds in-place to maintain `target_rate`.

- `networks.py`
  - `torchVPRSNN(nn.Module)` : Core unsupervised SNN model.
    - Constructor args: `n_in, n_exc, t_steps, beta_e, beta_i, thr_e_init, thr_i, a_plus, a_minus, tau_pre, tau_post, w_max, w_ei, w_ie, thr_eta, target_rate, device`
    - `forward(spk_in, do_stdp=True, monitor=False)` : runs temporal simulation over `T` timesteps; returns spike train `[T, B, n_exc]` and optionally an STDP `history` dict when `monitor=True`.

- `neuronal_assignments.py`
  - `get_training_spike_counts(net, encoder, loader, n_exc, n_classes, device)` : computes `S_R` matrix `[n_exc, n_classes]`.
  - `weighted_assignment_inference(S_Q, S_R, gamma=0.02)` : computes weighted-assignment similarity scores `[n_query, n_classes]`.
  - `probability_based_assignment(scores)` : row-wise min-max normalization → PDF-like probabilities.

- `vprsnn_evaluation.py`
  - `get_standard_assignments(net, encoder, loader, n_exc, n_classes, device)` : standard neuron-to-place assignment (argmax average rate per class); returns `assignments, avg_rates`.
  - `evaluate_vpr(net, encoder, loader, assignments, n_classes, device)` : runs inference across a loader and computes accuracy, similarity and probability matrices, and returns `accuracy, preds, targets, similarity_matrix, prob_matrix, S_Q`.
  - Plotting helpers: `plot_distance_matrix`, `plot_pr_curve`, `plot_recall_at_n`, `plot_weights`, `plot_neuron_assignments`, `visualize_qualitative_results`.
  - Dataset helpers: `NordlandDataset(dir_path, transform)`, `PatchNormalization`, `CLAHE`, and `get_nordland_loaders(train_path, test_path, batch_size, max_samples, shuffle_train)`.

- `snn_model.py`
  - `main()` : End-to-end experiment driver (data loading, STDP training, assignment, evaluation, and visualization).
  - The `params` dictionary near `main()` controls hyperparameters and directories.
  - **Automatic weight saving**: After every training run, `main()` saves a checkpoint to `weights/` at the project root. The filename encodes key characteristics (dataset, architecture, STDP params, seed, timestamp). Each checkpoint dict contains:
    - `state_dict`: Full model state dict (compatible with `torchVPRSNN.load_state_dict()`).
    - `params`: Dict of hyperparameters (dataset, n_in, n_exc, resolution, epochs, t_steps, wta_mode, enable_homeostasis, enable_weight_norm, seed, a_plus, a_minus, thr_e_init, target_rate, thr_eta, descriptor, and optionally room).
    - `timestamp`: Creation timestamp string (`YYYYMMDD_HHMMSS`).
  - **CLI flags**:
    - `--save-model <path>`: Save a named checkpoint (includes assignments and num_classes alongside state_dict).
    - `--load-model <path>`: Load a checkpoint and skip STDP training.


- `utils.py`
  - `Logger(filename)` : redirects `stdout` to a file and terminal.
  - `plot_stdp_monitor(history, epoch, save_dir)` : visualizes STDP traces and accumulated dW for a monitored sample.

- `metrics.py`
  - `createPR(S_in, GThard, GTsoft=None, matching='multi', n_thresh=100)` : computes precision and recall arrays for retrieval evaluation.
  - `recallAt100precision(S_in, GThard, GTsoft=None, matching='multi', n_thresh=100)` : recall@100% precision.
  - `recallAtK(S, GT, K=1)` : recall@K.

Notes
- Most functions assume CPU/GPU tensors behave consistently; many evaluation utilities convert data to/from NumPy for plotting and metric computation.
- See `docs/USAGE.md` for example run commands and `snn_model.py` for a concrete experiment configuration.
