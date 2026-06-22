# Installation

Minimal supported platform: Linux / macOS / Windows (WSL).
Python 3.8+ recommended.

Core dependencies (inferred from imports):
- torch (PyTorch)
- torchvision
- snntorch
- numpy
- matplotlib
- opencv-python (or opencv-contrib-python)
- scikit-learn
- pillow
- seaborn
- tqdm

Optional tooling:
- conda (recommended for cleaner environments)
- pip

Recommended conda one-liner (creates environment and installs common deps):

```bash
conda create -n vpr_snn python=3.10 -y
conda activate vpr_snn
conda install pytorch torchvision -c pytorch -c conda-forge -y
conda install numpy matplotlib scikit-learn seaborn -c conda-forge -y
pip install snntorch opencv-contrib-python tqdm
```

If you prefer pip inside an existing environment:

```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn snntorch opencv-contrib-python tqdm
```

Notes:
- `snntorch` is installed via `pip` (PyPI). Some configurations require a specific PyTorch build; consult the snntorch docs if you encounter incompatibilities.
- GPU support requires a CUDA-enabled PyTorch installation. Follow the official PyTorch instructions for your CUDA version.
