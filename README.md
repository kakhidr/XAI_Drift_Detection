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