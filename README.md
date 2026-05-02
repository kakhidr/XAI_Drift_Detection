# 🔍 XAI Drift Detection

**Detecting Adversarial Manipulation via Attribution Drift in Intrusion Detection Systems**

*Developed by **Karim Khidr** — MSc Cybersecurity, Privacy and Trust, [SETU](https://www.setu.ie/) (South East Technological University)*

---

## Overview

This project investigates whether **Explainable AI (XAI)** can detect **adversarial attacks** against machine-learning-based Intrusion Detection Systems (IDS). Rather than relying solely on model predictions, it analyses how **feature attributions** (the explanations behind predictions) shift when inputs are adversarially perturbed — a phenomenon we call **attribution drift**.

### The Core Idea

1. Train a neural network IDS on real network traffic data
2. Generate adversarial examples that attempt to fool the classifier
3. Compute feature attributions (explanations) for both clean and adversarial inputs
4. Measure the **drift** between explanation pairs using distance metrics
5. Evaluate whether drift can reliably distinguish adversarial from clean traffic

If attributions shift significantly under attack, drift-based detection can complement traditional IDS — even when the model's final prediction doesn't change.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (tested with 3.13)
- **pip** package manager
- ~2 GB disk space for dependencies (PyTorch, etc.)

### 1. Clone the Repository

```bash
git clone https://github.com/kakhidr/XAI_Drift_Detection.git
cd XAI_Drift_Detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

> **⚠️ Windows Note:** If you use the Microsoft Store Python and encounter long-path errors during PyTorch installation, create the venv at a shorter path (e.g., `python -m venv C:\venv\xai`) to avoid the Windows 260-character path limit.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch (CPU), scikit-learn, Captum, SHAP, Streamlit, and all other required packages.

### 4. Download Datasets

Datasets are **not included** in this repository. Download one or both:

| Dataset | Description | Download Link |
|---------|-------------|---------------|
| **CICIDS 2018** | Network-based intrusion detection benchmark with labelled flows for DoS, DDoS, Brute Force, Infiltration, and more | [UNB CICIDS2018 Page](https://www.unb.ca/cic/datasets/ids-2018.html) — follow the AWS download links |
| **BETH** | Real-world host-based intrusion detection data from honeypot systems | [Kaggle BETH Dataset](https://www.kaggle.com/datasets/katehighnam/beth-dataset) — requires free Kaggle account |

After downloading, place the CSV files in the appropriate folders:

```
data/
├── cicids2018/
│   ├── DoS attacks-GoldenEye.csv
│   ├── DDoS attack-HOIC.csv
│   └── ... (any CICIDS2018 CSV files)
└── beth/
    └── beth.csv (or any BETH CSV files)
```

> **Tip:** You can also upload CSV files directly through the web interface — no need to place them manually.

### 5. Launch the Web Interface

```bash
streamlit run app.py
```

This opens a browser at `http://localhost:8501` with the full interactive pipeline.

---

## 🧭 Using the Web Interface

### Sidebar Configuration

| Setting | Options | Description |
|---------|---------|-------------|
| **Dataset** | `cicids2018` / `beth` | Which dataset family to use |
| **CSV File** | (auto-detected) | Select from available files, or upload a new one |
| **Attack Type** | `fgsm` / `pgd` / `both` | Adversarial attack method |
| **XAI Method** | `ig` / `shap` / `both` | Explainability technique |
| **Sample Size** | 500 / 1000 / 3000 | Number of evaluation samples (affects speed & robustness) |
| **Epsilon (ε)** | 0.001 – 1.0 | Perturbation budget (larger = stronger attack) |
| **PGD Iterations** | 1 – 100 | Number of iterative gradient steps (PGD only) |
| **Epochs** | 1 – 100 | Maximum training epochs (early stopping may halt earlier) |
| **Batch Size** | 64 – 1024 | Training batch size |
| **Learning Rate** | 0.00001 – 0.1 | Optimizer learning rate |

### Results Dashboard

After clicking **🚀 Run Pipeline**, the results page shows:

- **⏱️ Timing Breakdown** — wall-clock time per stage with throughput analysis and hardware commentary
- **📈 Training History** — loss curves with early-stopping and overfitting detection
- **📋 Metrics Summary** — AUC scores per attack × XAI × drift metric, with per-config interpretations
- **📉 ROC Curves** — visual detection performance for each configuration
- **📊 Drift Histograms** — clean vs adversarial attribution distributions
- **📥 Download** — all outputs as a ZIP file for thesis inclusion

All captions are **dynamically generated** based on your specific configuration and results — they reference your chosen ε, sample size, attack type, and the actual AUC/timing values.

### Recommended Configurations

| Purpose | Attack | XAI | Samples | Notes |
|---------|--------|-----|---------|-------|
| **Quick test** | fgsm | ig | 500 | ~30 seconds, verify setup works |
| **Balanced run** | both | both | 1000 | ~2–5 minutes, good coverage |
| **Thesis figures** | both | both | 3000 | ~10+ minutes, most robust results |
| **Epsilon sweep** | fgsm | ig | 1000 | Run multiple times with ε = 0.001, 0.01, 0.1, 0.3 |

---

## 📁 Project Structure

```
XAI_Drift_Detection/
├── app.py                          # Streamlit web interface (main entry point)
├── configs/
│   └── experiment.yaml             # Base configuration (hyperparameters, paths)
├── src/
│   ├── data/
│   │   └── loader.py               # Dataset loading, preprocessing, scaling
│   ├── models/
│   │   └── mlp.py                  # MLP classifier with early stopping
│   ├── attacks/
│   │   ├── fgsm.py                 # Fast Gradient Sign Method
│   │   └── pgd.py                  # Projected Gradient Descent
│   ├── explain/
│   │   ├── ig.py                   # Integrated Gradients (Captum)
│   │   └── shap_explainer.py       # SHAP DeepExplainer
│   ├── drift/
│   │   └── metrics.py              # Cosine, Euclidean, KL divergence
│   ├── eval/
│   │   ├── roc.py                  # ROC-AUC computation and plotting
│   │   └── plots.py                # Drift histogram generation
│   ├── pipeline/
│   │   └── run_experiment.py       # End-to-end orchestrator
│   └── utils/
│       └── timing.py               # Per-stage timing infrastructure
├── notebooks/
│   └── min_feasibility.ipynb       # Original feasibility notebook (reference only)
├── main.py                         # Legacy monolithic script (reference only)
├── drift_plots/                    # Reference plots from early experiments
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

> **Note:** `main.py` and `notebooks/` are retained for reference. All active logic lives in `src/` and is orchestrated by `app.py`.

---

## ⚙️ Technical Details

### Pipeline Stages

| # | Stage | Description |
|---|-------|-------------|
| 1 | **Data Loading** | Load CSV, detect label columns, binarise labels (Benign=0, Attack=1), handle NaN/Inf, stratified train/test split, StandardScaler |
| 2 | **Model Training** | MLP classifier (configurable hidden layers), early stopping on validation loss, LR scheduler, progress callbacks |
| 3 | **Eval Subset** | Balanced selection of correctly-classified samples (50% benign, 50% attack) for fair evaluation |
| 4 | **Adversarial Attack** | FGSM (single gradient step) and/or PGD (iterative projected gradient descent with ε-ball clipping) |
| 5 | **XAI Attribution** | Integrated Gradients (via Captum) and/or SHAP DeepExplainer on clean + adversarial inputs |
| 6 | **Drift Measurement** | Cosine similarity, Euclidean distance, KL divergence between clean and adversarial attribution vectors |
| 7 | **ROC Evaluation** | Binary classification: clean samples (score=0) vs adversarial (score=drift distance), compute AUC |

### Drift Metrics

- **Cosine Distance** — measures angular change in attribution direction (insensitive to magnitude)
- **Euclidean Distance** — measures absolute change in attribution vectors
- **KL Divergence** — measures distributional shift between attribution probability distributions

### Preserved vs Flipped Predictions

Only samples where the model's prediction **doesn't change** after attack ("preserved predictions") are used for attribution comparison. This isolates the effect of adversarial perturbation on explanations rather than on outcomes.

### Reproducibility

All experiments use a fixed random seed (default: 42) for deterministic results across runs. The seed controls data splits, model initialisation, and evaluation sampling.

---

## 🔬 Research Context

### Research Questions

1. Can attribution drift reliably detect adversarial manipulation in IDS?
2. Which XAI method (IG vs SHAP) is more sensitive to adversarial perturbation?
3. Which drift metric (Cosine, Euclidean, KL) provides the best detection performance?
4. How does detection performance vary across datasets (network-based vs host-based)?
5. How does perturbation strength (ε) affect both attack success and detection capability?

### Adversarial Attack Methods

- **FGSM** (Fast Gradient Sign Method) — single-step, computationally cheap, baseline attack
- **PGD** (Projected Gradient Descent) — iterative, stronger, more realistic threat model

### Explainability Methods

- **Integrated Gradients (IG)** — gradient-based, follows a path from baseline to input
- **SHAP (DeepExplainer)** — Shapley-value-based, uses background dataset for reference

---

## ⚠️ Known Limitations

- **Hardware constraints** — large sample sizes (3000+) with both attacks and both XAI methods can be computationally expensive. GPU acceleration is recommended for production-scale experiments.
- **CICIDS Hulk subset** — not supported due to its large size and memory requirements.
- **CPU-only PyTorch** — the default installation uses CPU. For GPU support, install the appropriate CUDA version of PyTorch manually.
- **Windows long paths** — PyTorch installation may fail on Windows if the Python path is too long. Use a short venv path as described in the setup instructions.

---

## 📄 License

This project is part of academic dissertation research at SETU. Please cite appropriately if using any part of this work.

---

## 👤 Author

**Karim Khidr**
MSc Cybersecurity, Privacy and Trust
[South East Technological University (SETU)](https://www.setu.ie/)

---

*© 2025 Karim Khidr — All rights reserved*
