# XAI Drift Detection
## Detecting Adversarial Manipulation via Attribution Drift
### Overview

This project investigates whether drift in model explanations can be used to detect adversarial manipulation and distribution shifts in intrusion detection systems (IDS).

Rather than relying only on model predictions, the approach analyzes changes in feature attributions and evaluates their effectiveness using ROC-AUC metrics.

### Objectives
Primary Objective

To determine whether explanation drift can act as a reliable and dataset-agnostic signal of adversarial manipulation.

## Secondary Objective

To evaluate the consistency and effectiveness of explainability methods when used together:

Integrated Gradients (IG)
SHAP
Datasets

The project is designed to work with:

CICIDS2018 (network-based intrusion detection)
BETH dataset (host-based security dataset)
Methodology
Pipeline

Data flows through the following stages:
**Data → Model → Attack → Explainability → Drift Measurement → ROC Evaluation**

### Core Idea

The project computes drift between feature attributions before and after perturbation using:

Cosine distance
Euclidean distance

Both Integrated Gradients and SHAP are applied in every experiment, and their attribution outputs are used jointly for drift analysis.

**Adversarial settings supported:**


FGSM only
PGD only
Combined FGSM and PGD
### Project Structure

**The repository is organized into modular components covering:**

Data loading and preprocessing
Model training
Adversarial attack generation
Explainability (IG and SHAP)
Drift computation
Evaluation using ROC-AUC
End-to-end experiment pipeline

All implementation logic resides in the src directory. Notebooks are for reference only.

Running the Project

Experiments are executed through a unified pipeline.

Configuration supports:

**Dataset selection (CICIDS2018 or BETH)**
Attack setting:
none
fgsm
pgd
both

Both IG and SHAP are always executed as part of the pipeline.

Data Setup

Datasets are not included in this repository.

CICIDS2018 data should be placed in: data/cicids2018/
BETH data should be placed in: data/beth/
Loader Requirements

**The data loader:**

Converts labels to binary (Benign = 0, Attack = 1)
Uses numeric features only
Handles missing and infinite values
Performs stratified train/test split
Applies standard scaling (fit on training data only)
Outputs

**Each experiment run generates a results directory containing:**

Configuration snapshot
Metrics summary
ROC plots for:
Cosine drift
Euclidean drift
Metrics

**The evaluation includes:**

ROC-AUC for cosine drift
ROC-AUC for Euclidean drift
Mean cosine drift
Mean Euclidean drift
Definition of Done

**A complete implementation must:**

Run without errors
Work for both datasets
Execute IG and SHAP in every run
Support all attack configurations
Generate required outputs and ROC curves
Be reproducible using a fixed random seed
Limitations
The pipeline does not support the Hulk subset of the CICIDS2018 dataset due to its large size and memory constraints.
Reproducibility

All experiments are deterministic and controlled by a fixed seed defined in the configuration.

**Contribution Guidelines**
Use feature branches for development
Submit pull requests linked to tasks
Ensure the full pipeline runs locally before submission
Keep production code within the src directory
Do not include notebook logic in core modules
Research Context

Adversarial manipulation can alter model decisions without obvious statistical changes in input data.

**This project evaluates whether explanation drift:**

Detects adversarial manipulation
Generalizes across datasets
Remains consistent when combining multiple explainability methods
Notes
All core logic must reside in the src directory
Notebooks are for experimentation only
The pipeline must run end-to-end without manual intervention