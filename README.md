# XAI Drift Detection
Detecting Adversarial Manipulation via Attribution Drift
=========================================================

## Project Overview

This project investigates whether adversarial manipulation or distribution shift in intrusion detection systems (IDS) can be detected by measuring drift in model explanations.

The project evaluates:

**Datasets**
- **CICIDS2018** (network-based intrusion detection)
  - Link :  https://www.unb.ca/cic/datasets/ids-2018.html
- **BETH** (host-based security dataset)
  - Link : https://www.kaggle.com/datasets/katehighnam/beth-dataset
    
**Explainability Methods**
- **Integrated Gradients (IG)**
- **SHAP**

**Core idea:** instead of detecting drift using predictions alone, we detect drift in **feature attributions** and evaluate separability using **ROC-AUC**.

---

## Research Questions

**Primary Question**
> Can explanation drift serve as a reliable and dataset-agnostic signal of adversarial manipulation?

**Secondary Question**
> Which XAI method (IG or SHAP) provides more stable and discriminative drift signals?

---

## High-Level Architecture

**Pipeline Flow**
Data → Model → Attack/Shift → Explainability → Drift Metrics → ROC Evaluation

**Module Structure**
- `src/data/loader.py` → Dataset loading (CICIDS2018 + BETH)
- `src/models/mlp.py` → Neural network architecture and training
- `src/attacks/fgsm.py`, `src/attacks/pgd.py` → Adversarial perturbations
- `src/explain/ig.py` → Integrated Gradients
- `src/explain/shap.py` → SHAP
- `src/drift/metrics.py` → Cosine & Euclidean drift metrics
- `src/eval/roc.py` → ROC-AUC evaluation
- `src/pipeline/run_experiment.py` → Experiment orchestrator

All experiment logic must live in `src/`.  
Notebooks are for reference only.

---

## Repository Contract

This repository is intentionally scaffolded.

The student team must implement the required modules so that the following command runs successfully:

`python3 -m src.pipeline.run_experiment --config configs/experiment.yaml`

**The configuration must allow switching between:**
- Dataset: cicids2018 or beth
- XAI method: ig or shap
- Attack type: none, fgsm, or pgd

No notebook code should be required for a full experiment run.

⸻

## Setup

`python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt`

**Required libraries include:**
- torch
- captum
- shap
- numpy
- pandas
- scikit-learn
- matplotlib
- pyyaml

⸻

## Data Requirements

This repository does **NOT** include datasets.

**CICIDS2018**

Place CSV files in:

`data/cicids2018/`

**BETH**

Place BETH dataset files in:

`data/beth/`

The loader must:
- Support dataset selection via config
- Convert labels to binary (Benign → 0, Attack → 1)
- Use numeric features only
- Handle NaN / Inf values
- Perform stratified train/test split
- Apply StandardScaler (fit on train, transform test)

Loader must be dataset-agnostic.

⸻

## Definition of Done

A successful experiment run MUST:
1. Execute without import or runtime errors.
2. Work for all combinations:
   - CICIDS2018 + IG
   - CICIDS2018 + SHAP
   - BETH + IG
   - BETH + SHAP
3. Create:

`results/logs/<run_id>/`

4. Save:
   - config_snapshot.yaml
   - metrics_summary.json
5. Save ROC plots for:
   - Cosine drift
   - Euclidean drift
6. Be reproducible with fixed seed.

⸻

## Output Schema

`metrics_summary.json` must contain:

`{
  "dataset": "cicids2018 | beth",
  "xai_method": "ig | shap",
  "attack_type": "none | fgsm | pgd",
  "auc_cosine": float,
  "auc_euclidean": float,
  "mean_cosine_drift": float,
  "mean_euclidean_drift": float
}`

Any deviation from this schema is considered incomplete.

⸻

## Contribution Guidelines
- Create a feature branch per issue.
- Open Pull Request referencing the issue.
- PR must pass full experiment run locally.
- No notebook logic inside production modules.
- Code must be deterministic using cfg.run.seed.

⸻

## Research Context

Adversarial manipulation can alter model decisions without obvious statistical changes.

This project evaluates whether explanation drift:
	•	Detects adversarial manipulation,
	•	Generalizes across datasets,
	•	Remains stable across XAI methods.

This repository supports MSc-level research experimentation and comparative evaluation.

