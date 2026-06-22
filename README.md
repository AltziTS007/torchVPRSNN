# VPR-SNN: Unsupervised Spiking Neural Network for Visual Place Recognition

This repository contains a PyTorch-based implementation of an unsupervised Spiking Neural Network (SNN) tailored for Visual Place Recognition (VPR). The architecture is inspired by the Diehl & Cook model, utilizing Spike-Timing-Dependent Plasticity (STDP) for learning and a Winner-Take-All (WTA) mechanism for neuron specialization.

## Architecture

The system is built with a highly modular architecture, encapsulating different SNN dynamics and evaluation metrics:

- **Input Encoding** (`encoders.py`): Converts normalized image pixels into Poisson spike trains (Rate-based encoding).
- **Core Network** (`networks.py`, `mechanisms.py`): Implements the two-layer SNN (Excitatory and Inhibitory populations) featuring:
  - Hard Winner-Take-All (WTA) competition.
  - Weight-dependent STDP (without backpropagation).
  - Homeostatic threshold adaptation to balance neuron firing rates.
- **Assignment & Inference** (`neuronal_assignments.py`): Matches trained neurons to specific places (classes) using:
  - Standard (Highest Spike Count) Assignments.
  - Weighted Neuronal Assignments.
  - Probability-based Assignments.
- **Evaluation & Metrics** (`vprsnn_evaluation.py`, `metrics.py`): Evaluates cross-season place recognition tasks (e.g., Spring/Fall vs. Summer on the Nordland dataset), computing Precision, Recall (P@100R, R@100P), Area Under the Precision-Recall Curve (AUC-PR), and generating diagnostic visualizations (Distance Matrices, PR Curves).

## Installation

The project is designed to run within a Conda environment. 

If you do not have the environment yet, you can create it and install all dependencies in one go:

```bash
conda create -n snntorch python=3.10 pytorch torchvision numpy matplotlib scipy scikit-learn pandas tqdm -c pytorch -c conda-forge -y && conda activate snntorch && pip install snntorch opencv-contrib-python seaborn
```

If you prefer to use an existing environment, you can install the required packages via `pip`:

```bash
pip install torch torchvision numpy matplotlib scipy scikit-learn pandas tqdm snntorch opencv-contrib-python seaborn
```

## Dataset Preparation

The system is configured to use the **Nordland** dataset by default. Check `snn_model.py` and ensure the dataset is located in the working directory as follows:
- **Training (Reference)**: `nordland_clean/data/spring` and `nordland_clean/data/fall`
- **Testing (Query)**: `nordland_clean/data/summer`

## Execution

To run the end-to-end experiment—which includes data loading, STDP training, neuron-to-place assignment, and comprehensive evaluation—execute the main script:

```bash
conda activate snntorch
python snn_model.py
```

### Experiment Outputs

During execution, the script automatically creates a timestamped results directory (e.g., `results_YYYY-MM-DD_HH-MM-SS/`). Inside this directory, the following artifacts are saved:

- **`output.log`**: A complete dump of the console output, including hardware telemetry (latency, energy, FLOPs).
- **`hyperparameters.json`**: A record of the hyperparameters configured for the run.
- **`weights_history/`**: Visualizations of the excitatory weights saved at regular intervals (e.g., every 10 epochs).
- **`stdp_viz/`**: Plots monitoring STDP history and learning dynamics.
- **Evaluation Directories** (e.g., `Standard/`, `Weighted/`, `Weighted+Prob/`): Contain method-specific performance plots including Distance Matrices, Precision-Recall Curves, Recall@N plots, and illustrative qualitative results comparing matched locations.
- **Inference Profile**: Generates comprehensive hardware telemetry reporting SNN simulation latency, Sequence Aggregation overhead, sustained FPS, and energy draw per query.

## Reproducing the Paper Results

This repository is structured to allow full reproducibility of the results reported in *thirteenth_draft.pdf*. Below are the exact commands required to reproduce each table and figure from a clean checkout. 

### 1. Table IV: Operational Envelope of Sequence Aggregation
This evaluates the robustness of the VPR system against velocity mismatches, reverse traversals, dropped frames, and route deviations.
- **Nordland Dataset**:
  ```bash
  python run_table_iv_real.py --data-dir nordland_clean
  ```
- **Oxford RobotCar Dataset**:
  ```bash
  python run_table_iv_oxford.py --data-dir ORC
  ```

### 2. Table V: Sequential Aggregation (R@100P vs. Window Size $k$)
This evaluates the impact of varying the sliding window size on overall recall performance.
```bash
python run_table_v.py --data-dir nordland_clean
```

### 3. Figure: Sliding Window (Chunked) Protocol Curves
To generate the performance degradation curves under varying spatial chunk sizes (comparing AUC-PR across datasets):
```bash
bash run_sliding_window_full_curve.sh
bash run_sliding_window_full_curve_oxford.sh
python plot_combined_auc.py
```
*Output: Generates `ablation_sliding_window_combined_auc.png`.*

### 4. Figure: Timestep (Latency/Energy) Ablation
To generate the Pareto frontier curves evaluating the optimal deployment sweet spot (latency vs. performance):
```bash
bash run_ablation_tsteps.sh
```
*Output: Generates `ablation_latency_vs_r100p.png` and `ablation_energy_vs_r100p.png`.*

### 5. 15-Seed Baseline Statistical Sweeps
To ensure reproducibility and publication-ready statistical significance, perform $N$-seed sweeps (e.g., 15 repetitions) and automatically extract the Mean ± Std Dev performance. This protocol establishes the robust 77.93% ± 3.97% mean R@100P baseline for the Probability-Based assignment (Table III).
```bash
bash run_max_samples_benchmark.sh
python extract_sweep_stats.py --start "YYYY-MM-DD HH:MM:SS" --end "YYYY-MM-DD HH:MM:SS"
```

### 6. Table VI: State Isolation (Within-System Ablation)
To reproduce the explicit ablation of the state-isolation mechanism (Temporal Spill-over vs. Healthy Reset):
```bash
bash run_ablation_state_isolation.sh
```

### 7. Impact of Neuronal Assignment Strategy (Boxplots)
To aggregate and visualize the raw `Standard`, `Weighted`, and `Weighted+Prob` table outputs across your multi-seed runs (Figure 1), use the table extraction pipeline:
```bash
python extract_tables_by_time.py --start "YYYY-MM-DD HH:MM:SS" --end "YYYY-MM-DD HH:MM:SS"
python plot_r100p_from_extracted_tables.py --in-file results/tables_...txt --out r100p_boxplot.png
```
*Output: Generates statistical boxplots comparing the assignment strategies and saves a `r100p_summary.txt` with exact mean values.*

> **Note**: Secondary exploratory scripts, old ablation logs, and raw sweeping utilities have been archived into the `secondary/` directory for reference, ensuring the root directory remains dedicated to the core reproducible pipeline.

## Documentation

Comprehensive documentation is available in the `docs/` folder. Key pages:

- `docs/INSTALL.md` — Installation and environment setup
- `docs/USAGE.md` — Quickstart, common run flows, and tips
- `docs/API.md` — API reference for main modules and functions
- `docs/CONTRIBUTING.md` — Contribution guidelines and project conventions

Open `docs/README.md` for an index of the documentation.
