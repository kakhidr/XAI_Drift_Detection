# Changelog

All notable changes to the **XAI Drift Detection** project are documented in this file.

> **Author:** Karim Khidr — MSc Cybersecurity, Privacy and Trust, SETU  
> This project is part of a dissertation investigating how Explainable AI (XAI) methods can detect adversarial drift in Intrusion Detection Systems.

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
