# Changelog

All notable changes to the **XAI Drift Detection** project are documented in this file.

> **Author:** Karim Khidr — MSc Cybersecurity, Privacy and Trust, SETU  
> This project is part of a dissertation investigating how Explainable AI (XAI) methods can detect adversarial drift in Intrusion Detection Systems.

---

## [1.4.1] — 2026-05-10

### Removed
- KL Divergence from pipeline (`run_experiment.py`) and Streamlit app — only Cosine and Euclidean drift metrics are now computed, plotted, and reported

---

## [1.4.0] — 2026-05-10

### Added
- **Research Plots Generator** (`legacy/research_plots.py`) — standalone script for generating publication-ready figures from experiment results, designed for MSc dissertation inclusion
  - Epsilon vs AUC line charts (Cosine + Euclidean combined)
  - Epsilon vs Mean Drift log-scale plots
  - IG vs SHAP grouped bar comparison
  - AUC heatmap summary across all configurations
  - Prediction flip rate vs perturbation strength
  - **Clean Explanation Stability vs Adversarial Drift** distribution histograms for FGSM and PGD (both Cosine and Euclidean)
  - Multi-metric ROC overlays per configuration
  - Top-K feature attribution shift bar charts
  - Space-saving combined 2×2 grid layouts for dissertation figures
  - Individual standalone figures for flexible placement
- Raw score caching (`raw_scores.npz`) — re-run pipeline once, regenerate plots instantly
- All figures exported as both PNG (300 DPI) and PDF (vector) for thesis embedding

### Removed
- KL Divergence from research plots (not required for dissertation analysis)

---

## [1.3.0] — 2026-05-06

### Added
- **Research-rigor metric evaluation** — ROC-AUC now compares adversarial attribution drift against clean-pair attribution drift instead of an artificial zero baseline
- Bootstrap 95% AUC confidence intervals and clean-percentile threshold metrics for drift ROC outputs
- Structured run provenance saved to `run_metadata.json`, including resolved config, dataset metadata, model metrics, package versions, device, seed, and evaluation subset details
- Structured `metrics_schema.json` separating model metrics, attack metrics, drift detection metrics, baseline metrics, and data metadata
- **Thesis-ready `experiment_summary.csv` export** — flat table with one row per attack × XAI × drift metric, including AUC, confidence intervals, clean/adversarial drift means, flip rate, preserved count, model accuracy, and baseline AUCs
- Research baselines:
  - prediction-confidence drift baseline
  - raw-input L2 perturbation baseline
  - random-attribution null baseline
- Streamlit pre-run warnings for risky configurations, including small/large ε, PGD alpha issues, low PGD iterations, SHAP-heavy runs, large sweeps, and CPU-heavy settings
- Streamlit post-run result quality checks with recommended adjustments for low model accuracy, high flip rate, too few preserved samples, low strict-baseline AUC, and wide bootstrap confidence intervals
- Unit and smoke tests covering data loading, evaluation subset sampling, drift metrics, ROC details, and the full synthetic FGSM + IG pipeline

### Changed
- Evaluation subset selection is now seeded, randomly sampled, class-balanced, and records requested vs actual counts
- Dataset loading now validates empty files, missing labels, single-class data, severe class imbalance, missing numeric features, NaN/Inf values, and dropped non-numeric columns
- `run_pipeline()` and `run_epsilon_sweep()` keep their existing call signatures but return `run_metadata`, `metrics_schema`, and `summary_path`
- Streamlit result captions now explain strict clean-baseline AUC instead of assuming AUC should always increase with ε
- README updated with clean-baseline AUC interpretation, configuration warnings, provenance files, and `experiment_summary.csv`
- Matplotlib/Fontconfig cache handling now uses writable temp directories to avoid noisy import/runtime warnings in constrained environments

---

## [1.2.0] — 2026-05-04

### Added
- **Epsilon Sweep Mode** — new run mode in the Streamlit app that sweeps over multiple ε values (2–10) in a single experiment
- `run_epsilon_sweep()` function in `src/pipeline/run_experiment.py` — trains model once, reuses across all ε values for efficiency
- Sweep sidebar UI: run mode toggle, ε count slider (2–10), individual ε value inputs with sensible defaults
- Sweep results dashboard:
  - Summary table with all metrics per ε value
  - Drift vs ε line plots (cosine, euclidean, KL divergence)
  - AUC vs ε subplots showing detection performance trends
  - Flip rate vs ε chart showing attack success rate
- Dynamic captions for sweep results interpreting trends, best/worst configurations, and sensitivity analysis
- README updated with full Epsilon Sweep Mode documentation

---

## [1.1.0] — 2026-05-02

### Added
- **Streamlit Web Interface** (`app.py`) — full interactive pipeline replacing the monolithic script approach
- Complete implementation of all 11 `src/` modules (previously stubs):
  - `src/data/loader.py` — dataset loading, preprocessing, train/test split, feature scaling
  - `src/models/mlp.py` — MLP classifier with configurable architecture and early stopping
  - `src/attacks/fgsm.py` — Fast Gradient Sign Method adversarial attack
  - `src/attacks/pgd.py` — Projected Gradient Descent adversarial attack
  - `src/xai/integrated_gradients.py` — Integrated Gradients attribution method
  - `src/xai/shap_explainer.py` — SHAP (DeepExplainer) attribution method
  - `src/metrics/drift.py` — Drift metrics: cosine similarity, euclidean distance, KL divergence
  - `src/metrics/roc.py` — ROC-AUC computation for drift-based detection
  - `src/visualization/plots.py` — ROC curves and drift histogram generation
  - `src/pipeline/run_experiment.py` — Pipeline orchestrator coordinating all stages
  - `src/utils/timing.py` — Stage-level timing with throughput analysis
- **Dynamic captions** throughout the results dashboard:
  - Timing breakdown with throughput metrics and hardware pressure warnings
  - Training history with early-stopping and overfitting detection
  - Metrics interpretation referencing actual AUC values and ε settings
  - ROC curve and histogram commentary adapting to configuration
- **CSV file uploader** — users upload their own dataset CSV files directly in the sidebar
- **Homepage introduction** with:
  - Project explanation and navigation guide
  - Dataset download links (CICIDS2017 and BETH)
  - Pipeline stages reference table
  - Usage tips for dissertation work
- **Author attribution** — header, introduction, and footer credit Karim Khidr, MSc Cybersecurity, Privacy and Trust, SETU
- **Comprehensive README** with setup guide, dataset links, usage documentation, project structure, technical details, and research context

### Changed
- `requirements.txt` — replaced 125-line pip freeze with clean 12-dependency minimal list
- `.gitignore` — added patterns for venv, IDE files, generated outputs, and OS metadata

### Removed
- 12 stale root-level `roc_*.png` files from git tracking

---

## [1.0.1] — 2026-05-02

### Changed
- **Legacy file organisation** — moved `main.py` and `notebooks/` into `legacy/` folder via `git mv`
- README updated to document legacy files as initial feasibility trials that helped build the pipeline
- Legacy files retained as standalone reference (each runs independently with its own dependencies)

---

## [1.0.0] — 2026-04-24

### Added
- `main.py` — monolithic pipeline script (1,136 lines) consolidating notebook logic into a single executable
- `.gitignore` — standard Python/ML ignores
- Cleaned project structure with proper metadata

### Changed
- README refactored for clarity with dataset links and project details

---

## [0.1.0] — 2026-02-16

### Added
- **Initial repository setup** — project scaffolding for XAI drift detection research
- `src/` module structure with stub implementations (`raise NotImplementedError`)
- `configs/experiment.yaml` — base experiment configuration
- `notebooks/min_feasibility.ipynb` — initial feasibility experiments (epsilon sweeps, drift analysis)
- `README.md` — initial project description and research context

---

## Version Summary

| Version | Date | Milestone |
|---------|------|-----------|
| **0.1.0** | 2026-02-16 | Initial repo scaffolding and feasibility notebook |
| **1.0.0** | 2026-04-24 | Monolithic `main.py` pipeline script |
| **1.0.1** | 2026-05-02 | Legacy file reorganisation |
| **1.1.0** | 2026-05-02 | Full Streamlit web UI with dynamic captions |
| **1.2.0** | 2026-05-04 | Epsilon Sweep mode for multi-ε analysis |
| **1.3.0** | 2026-05-06 | Research-rigor evaluation, provenance, warnings, and thesis summary export |
