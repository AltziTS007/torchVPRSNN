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
- **Evaluation & Metrics** (`vprsnn_evaluation.py`, `metrics.py`): Evaluates cross-season place recognition tasks (e.g., Spring/Fall vs. Summer on the Nordland dataset), computing Precision, Recall (P@100R, R@100P), and generating diagnostic visualizations (Distance Matrices, PR Curves).

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

- **`output.log`**: A complete dump of the console output.
- **`hyperparameters.json`**: A record of the hyperparameters configured for the run.
- **`weights_history/`**: Visualizations of the excitatory weights saved at regular intervals (e.g., every 10 epochs).
- **`stdp_viz/`**: Plots monitoring STDP history and learning dynamics.
- **Evaluation Directories** (e.g., `Standard/`, `Weighted/`, `Weighted+Prob/`): Contain method-specific performance plots including Distance Matrices, Precision-Recall Curves, Recall@N plots, and illustrative qualitative results comparing matched locations.
