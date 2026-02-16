# XAI_Drift_Detection
DETECTING ADVERSARIAL MANIPULATION VIA ATTRIBUTION DRIFT.
=========================================================

# Explanation Drift in IDS (CICIDS2018)

Goal: quantify how explanations (Integrated Gradients) change under distribution shift / adversarial stress, and evaluate explanation-drift separability using ROC-AUC.

## Repo Contract
This repo is intentionally scaffolded. Student team must implement the stubs in `src/` to satisfy the end-to-end run.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


# XAI Drift Detection
Detecting Adversarial Manipulation via Attribution Drift

---

# Explanation Drift in IDS (CICIDS2018)

## 1. Project Objective

This project investigates whether adversarial manipulation or distribution shift in intrusion detection systems (IDS) can be detected by measuring drift in model explanations.

Specifically, we:

- Train a neural network classifier on CICIDS2018.
- Generate feature attributions using Integrated Gradients (IG).
- Apply adversarial perturbations (FGSM / PGD) or distribution shift.
- Quantify attribution drift using cosine and Euclidean distance.
- Evaluate separability of clean vs shifted explanations using ROC-AUC.

The central research question:

> Can explanation drift serve as a reliable signal of adversarial manipulation?

---

## 2. High-Level Architecture

Pipeline flow:

Data → Model → Attack/Shift → Explainability → Drift Metrics → ROC Evaluation

Module mapping:

- `src/data/loader.py` → Dataset loading and preprocessing
- `src/models/mlp.py` → Neural network architecture and training
- `src/attacks/fgsm.py`, `src/attacks/pgd.py` → Adversarial perturbations
- `src/explain/ig.py` → Integrated Gradients attribution
- `src/drift/metrics.py` → Attribution drift computation
- `src/eval/roc.py` → ROC-AUC evaluation and plotting
- `src/pipeline/run_experiment.py` → Experiment orchestrator

All experiment logic must reside in `src/`.
Notebooks are for reference only.

---

## 3. Repository Contract

This repository is intentionally scaffolded.

Student contributors must implement the required modules so that the following command runs successfully:

```bash
python3 -m src.pipeline.run_experiment --config configs/experiment.yaml
```

No notebook logic should be required for a full experiment run.

---

## 4. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Data Requirements

This repository does NOT include CICIDS2018 data.

Place CSV files in:

```
data/cicids2018/
```

Expected format:

```
data/
  cicids2018/
    *.csv
```

The loader must:
- Convert labels to binary (Benign → 0, Attack → 1)
- Use numeric features only
- Handle NaN / Inf values
- Perform stratified train/test split
- Apply StandardScaler (fit on train, transform test)

---

## 6. Definition of Done

A successful experiment run MUST:

1. Execute without import or runtime errors.
2. Create a new directory in:
   `results/logs/<run_id>/`
3. Save:
   - `config_snapshot.yaml`
   - `metrics_summary.json`
4. Save ROC plots for:
   - Cosine drift
   - Euclidean drift
5. Be reproducible given a fixed seed.

---

## 7. Output Schema

`metrics_summary.json` must contain:

```json
{
  "auc_cosine": float,
  "auc_euclidean": float,
  "mean_cosine_drift": float,
  "mean_euclidean_drift": float
}
```

Any deviation from this schema is considered incomplete.

---

## 8. Modules to Implement

The following functions must be implemented by the team:

### Data
- `load_dataset(cfg)` → returns `X_train, X_test, y_train, y_test`

### Model
- `build_mlp(cfg, input_dim)`
- `train_model(model, X_train, y_train, cfg)`

### Explainability
- `compute_ig(model, X, y, cfg)`

### Drift Metrics
- `compute_cosine(attr_clean, attr_shifted)`
- `compute_euclidean(attr_clean, attr_shifted)`

### Attacks
- `fgsm_attack(model, X, y, epsilon)`
- `pgd_attack(model, X, y, cfg)`

### Evaluation
- `compute_roc(scores, out_dir, name)`

Each implementation must follow the function signatures exactly.

---

## 9. Contribution Guidelines

- Create a feature branch for each issue.
- Open a Pull Request referencing the issue.
- PR must pass a full experiment run locally.
- No experimental notebook logic in production modules.
- All code must be deterministic given `cfg.run.seed`.

---

## 10. Research Context

Adversarial manipulation can cause models to misclassify inputs while maintaining similar surface statistics.

Instead of detecting drift in predictions alone, this project investigates drift in feature attributions.

If explanations change significantly under adversarial stress, explanation drift may serve as a secondary detection signal.
# XAI Drift Detection
Detecting Adversarial Manipulation via Attribution Drift
=========================================================

# Explanation Drift in IDS (CICIDS2018 & BETH)

Goal: Quantify how model explanations change under adversarial manipulation or distribution shift across TWO datasets and TWO explainability methods.

This project evaluates whether explanation drift can serve as a detection signal across:

Datasets:
- CICIDS2018 (network intrusion dataset)
- BETH dataset (host-based security dataset)

Explainability Methods:
- Integrated Gradients (IG)
- SHAP (KernelSHAP or DeepSHAP depending on model compatibility)

---

## 1. Project Objective

This project investigates whether adversarial manipulation or distribution shift in intrusion detection systems (IDS) can be detected by measuring drift in model explanations.

Specifically, we:

- Train a neural network classifier on CICIDS2018 and BETH.
- Generate feature attributions using BOTH:
  - Integrated Gradients (IG)
  - SHAP
- Apply adversarial perturbations (FGSM / PGD) or distribution shift.
- Quantify attribution drift using cosine and Euclidean distance.
- Evaluate separability of clean vs shifted explanations using ROC-AUC.
- Compare drift behaviour across datasets and XAI methods.

Central research question:

> Can explanation drift serve as a reliable and dataset-agnostic signal of adversarial manipulation?

Secondary question:

> Does SHAP or Integrated Gradients provide more stable and discriminative drift signals?

---

## 2. High-Level Architecture

Pipeline flow:

Data → Model → Attack/Shift → Explainability (IG + SHAP) → Drift Metrics → ROC Evaluation

Module mapping:

- `src/data/loader.py` → Dataset loading and preprocessing (must support CICIDS2018 AND BETH)
- `src/models/mlp.py` → Neural network architecture and training
- `src/attacks/fgsm.py`, `src/attacks/pgd.py` → Adversarial perturbations
- `src/explain/ig.py` → Integrated Gradients attribution
- `src/explain/shap.py` → SHAP attribution
- `src/drift/metrics.py` → Attribution drift computation
- `src/eval/roc.py` → ROC-AUC evaluation and plotting
- `src/pipeline/run_experiment.py` → Experiment orchestrator

All experiment logic must reside in `src/`.
Notebooks are for reference only.

---

## 3. Repository Contract

This repository is intentionally scaffolded.

Student contributors must implement required modules so that the following command runs successfully for BOTH datasets and BOTH XAI methods:

```bash
python3 -m src.pipeline.run_experiment --config configs/experiment.yaml
```

The experiment configuration must allow switching between:

- Dataset: CICIDS2018 or BETH
- XAI method: IG or SHAP
- Attack type: none | fgsm | pgd

No notebook logic should be required for a full experiment run.

---

## 4. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure the following libraries are included:
- torch
- captum
- shap
- scikit-learn
- pandas
- numpy
- matplotlib

---

## 5. Data Requirements

This repository does NOT include datasets.

### CICIDS2018
Place CSV files in:

```
data/cicids2018/
```

### BETH Dataset
Place BETH dataset files in:

```
data/beth/
```

The loader must:
- Support dataset selection via config
- Convert labels to binary (Benign → 0, Malicious → 1)
- Use numeric features only
- Handle NaN / Inf values
- Perform stratified train/test split
- Apply StandardScaler (fit on train, transform test)

Loader design must be dataset-agnostic.

---

## 6. Definition of Done

A successful experiment run MUST:

1. Execute without import or runtime errors.
2. Run successfully for:
   - CICIDS2018 + IG
   - CICIDS2018 + SHAP
   - BETH + IG
   - BETH + SHAP
3. Create a new directory in:
   `results/logs/<run_id>/`
4. Save:
   - `config_snapshot.yaml`
   - `metrics_summary.json`
5. Save ROC plots for:
   - Cosine drift
   - Euclidean drift
6. Be reproducible given a fixed seed.

---

## 7. Output Schema

`metrics_summary.json` must contain:

```json
{
  "dataset": "cicids2018 | beth",
  "xai_method": "ig | shap",
  "attack_type": "none | fgsm | pgd",
  "auc_cosine": float,
  "auc_euclidean": float,
  "mean_cosine_drift": float,
  "mean_euclidean_drift": float
}
```

Any deviation from this schema is considered incomplete.

---

## 8. Modules to Implement

The following functions must be implemented by the team:

### Data
- `load_dataset(cfg)` → must support BOTH datasets

### Model
- `build_mlp(cfg, input_dim)`
- `train_model(model, X_train, y_train, cfg)`

### Explainability
- `compute_ig(model, X, y, cfg)`
- `compute_shap(model, X, y, cfg)`

### Drift Metrics
- `compute_cosine(attr_clean, attr_shifted)`
- `compute_euclidean(attr_clean, attr_shifted)`

### Attacks
- `fgsm_attack(model, X, y, epsilon)`
- `pgd_attack(model, X, y, cfg)`

### Evaluation
- `compute_roc(scores, out_dir, name)`

Each implementation must follow the function signatures exactly.

---

## 9. Contribution Guidelines

- Create a feature branch for each issue.
- Open a Pull Request referencing the issue.
- PR must pass a full experiment run locally.
- No experimental notebook logic in production modules.
- All code must be deterministic given `cfg.run.seed`.

---

## 10. Research Context

Adversarial manipulation can cause models to misclassify inputs while maintaining similar surface statistics.

Rather than relying only on prediction drift, this project evaluates explanation drift across:
- Multiple datasets
- Multiple XAI methods

By comparing IG and SHAP under identical stress conditions, we aim to evaluate robustness, stability, and discriminative power of explanation-based drift detection.

This repository supports MSc-level research experimentation and reproducible comparative evaluation.