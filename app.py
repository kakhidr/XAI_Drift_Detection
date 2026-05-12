"""
XAI Drift Detection — Streamlit Web Interface
Run with:  streamlit run app.py
"""
import os
import sys
import tempfile
import yaml
import pandas as pd
import streamlit as st

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "xai_drift_cache")
_MPL_DIR = os.path.join(tempfile.gettempdir(), "xai_drift_matplotlib")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.makedirs(_MPL_DIR, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", _MPL_DIR)

import matplotlib.pyplot as plt

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from src.data.loader import list_csv_files
from src.eval.plot_style import configure_plot_style, style_axis
from src.pipeline.run_experiment import run_pipeline, run_epsilon_sweep

configure_plot_style()


def build_pre_run_alerts(run_mode, attack_type, xai_method, max_eval, epsilon,
                         alpha, pgd_iters, eps_list=None, cuda_requested=False):
    errors = []
    warnings = []
    infos = []
    eps_values = eps_list if run_mode == "Epsilon Sweep" and eps_list else [epsilon]
    min_eps = min(eps_values)
    max_eps = max(eps_values)

    if any(e <= 0 for e in eps_values):
        errors.append("Epsilon values must be greater than 0. Non-positive ε will invalidate attack generation and plots.")
    if attack_type in ("pgd", "both") and alpha <= 0:
        errors.append("PGD alpha must be greater than 0.")
    if attack_type in ("pgd", "both") and pgd_iters < 1:
        errors.append("PGD iterations must be at least 1.")

    if min_eps < 0.005:
        warnings.append(
            f"Small ε detected (min ε={min_eps:g}). Attribution drift may be smaller than natural clean-sample variation, "
            "so strict clean-baseline AUC can be low even when the pipeline is working. Recommended adjustment: include "
            "moderate ε values such as 0.01, 0.02, or 0.05 in sweeps for comparison."
        )
    if max_eps >= 0.2:
        warnings.append(
            f"Large ε detected (max ε={max_eps:g}). Many predictions may flip, reducing the preserved-prediction sample "
            "used for attribution drift analysis. Recommended adjustment: keep high ε values for attack-success analysis, "
            "but include lower ε values for drift-focused ROC interpretation."
        )
    if attack_type in ("pgd", "both") and alpha > min_eps:
        warnings.append(
            f"PGD alpha ({alpha:g}) is larger than the smallest ε ({min_eps:g}). This can make PGD jump to the projection "
            "boundary immediately. Recommended adjustment: use alpha <= ε/4; for example ε=0.02 works well with alpha=0.005."
        )
    elif attack_type in ("pgd", "both") and alpha > min_eps / 2:
        warnings.append(
            f"PGD alpha ({alpha:g}) is large relative to ε ({min_eps:g}). Recommended adjustment: set alpha <= ε/4 for "
            "smoother optimization."
        )
    if attack_type in ("pgd", "both") and pgd_iters < 5:
        warnings.append(
            "PGD iterations below 5 may under-optimize the attack and make PGD curves hard to interpret. Recommended "
            "adjustment: use 10-20 iterations for thesis-quality PGD runs."
        )
    if xai_method in ("shap", "both") and max_eval >= 1000:
        warnings.append(
            "SHAP with 1000+ samples can be slow and memory-heavy, especially on CPU. Recommended adjustment: run IG first "
            "for exploration, then use SHAP on final smaller configurations."
        )
    if run_mode == "Epsilon Sweep" and attack_type == "both" and xai_method == "both":
        warnings.append(
            "Sweep + both attacks + both XAI methods generates many ROC plots and can take substantially longer. Recommended "
            "adjustment: sweep one attack/XAI pair first, then expand once settings look stable."
        )
    if run_mode == "Epsilon Sweep" and len(eps_values) >= 8:
        warnings.append(
            "Large epsilon sweeps create many output files. Recommended adjustment: use 4-6 ε values for quick iteration, "
            "then run the full sweep for final results."
        )
    if max_eval >= 3000:
        warnings.append(
            "3000 evaluation samples improves stability but can make IG/SHAP runs slow on CPU. Recommended adjustment: use "
            "500-1000 samples while tuning, then 3000 for final figures."
        )
    if not cuda_requested:
        infos.append("CUDA is disabled in config, so runs will use CPU even if a GPU is available.")

    infos.append(
        "AUC is evaluated against clean-pair attribution drift, not an artificial zero baseline. "
        "This is stricter and may produce lower AUC values than earlier versions."
    )
    return errors, warnings, infos


def build_result_quality_alerts(results):
    warnings = []
    infos = []
    schema = results.get("metrics_schema", {})
    model_metrics = schema.get("model_metrics", {})
    attack_metrics = schema.get("attack_metrics", {})
    drift_metrics = schema.get("drift_detection_metrics", {})

    test_accuracy = model_metrics.get("test_accuracy")
    if test_accuracy is not None and test_accuracy < 0.9:
        warnings.append(
            f"Model test accuracy is {test_accuracy:.3f}. Drift results may be unreliable because the base IDS is weak. "
            "Recommended adjustment: improve the classifier before interpreting drift results; try more epochs, check the "
            "label mapping/data file, and keep a stable learning rate such as 0.001."
        )

    for key, vals in attack_metrics.items():
        flip_rate = vals.get("flip_rate")
        n_preserved = vals.get("n_preserved")
        if flip_rate is not None and flip_rate > 0.5:
            warnings.append(
                f"{key}: flip rate is {flip_rate:.1%}. Attribution drift is computed only on preserved predictions, "
                "so the drift sample may no longer represent the full attack. Recommended adjustment: lower ε for this "
                "attack configuration, or for PGD reduce alpha/iterations; aim to keep most predictions preserved for "
                "drift-focused analysis."
            )
        if n_preserved is not None and n_preserved < 100:
            warnings.append(
                f"{key}: only {n_preserved} preserved samples remain. ROC/AUC estimates are statistically weak. "
                "Recommended adjustment: lower ε or increase Evaluation Sample Size; for thesis figures, prefer at least "
                "a few hundred preserved samples when runtime allows."
            )

    for key, metrics_by_name in drift_metrics.items():
        for metric_name, details in metrics_by_name.items():
            if metric_name == "random_attribution_null" or not isinstance(details, dict):
                continue
            auc_val = details.get("auc")
            if auc_val is not None and auc_val < 0.5:
                infos.append(
                    f"{key} {metric_name}: AUC={auc_val:.3f}. Under the strict clean-pair baseline, adversarial drift is "
                    "smaller than normal clean-sample attribution variation for this metric. Recommended adjustment: do not "
                    "treat this as a broken run; compare mean adversarial vs clean drift, try a wider ε sweep if needed, "
                    "and report it as conservative-baseline evidence."
                )
            ci = details.get("auc_ci_95")
            if ci:
                width = ci.get("high", 0) - ci.get("low", 0)
                if width > 0.2:
                    warnings.append(
                        f"{key} {metric_name}: bootstrap AUC CI is wide ({ci.get('low'):.3f}-{ci.get('high'):.3f}), "
                        "so the estimate has high uncertainty. Recommended adjustment: increase Evaluation Sample Size, "
                        "reduce attack strength if preserved samples are low, or repeat the run with more data."
                    )

    return warnings, infos


def show_alerts(title, errors=None, warnings=None, infos=None):
    errors = errors or []
    warnings = warnings or []
    infos = infos or []
    if not errors and not warnings and not infos:
        return
    st.subheader(title)
    for msg in errors:
        st.error(msg)
    for msg in warnings:
        st.warning(msg)
    for msg in infos[:5]:
        st.info(msg)
    if len(infos) > 5:
        with st.expander(f"Show {len(infos) - 5} additional notes"):
            for msg in infos[5:]:
                st.info(msg)

# ──────────────────────────────────────
# Page config
# ──────────────────────────────────────
st.set_page_config(page_title="XAI Drift Detection", layout="wide")
st.title("🔍 XAI Drift Detection Pipeline")
st.markdown(
    "Detect adversarial manipulation via attribution drift in intrusion detection systems.  \n"
    "*Developed by* ***Karim Khidr*** *— MSc Cybersecurity, Privacy and Trust, "
    "[SETU](https://www.setu.ie/) (South East Technological University)*"
)

# ──────────────────────────────────────
# Load base config
# ──────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "experiment.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

DATA_ROOT = cfg["data"]["root"]

# ──────────────────────────────────────
# Sidebar — Experiment Configuration
# ──────────────────────────────────────
st.sidebar.header("⚙️ Experiment Configuration")

# 1. Dataset selection
dataset = st.sidebar.selectbox("Dataset", ["cicids2018", "beth"])

# 2. CSV file selection + upload
csv_files = list_csv_files(DATA_ROOT, dataset)

uploaded = st.sidebar.file_uploader(
    f"Upload a CSV to `{dataset}/`", type=["csv"], key=f"upload_{dataset}"
)
if uploaded is not None:
    dest_dir = os.path.join(DATA_ROOT, dataset)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, uploaded.name)
    with open(dest_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success(f"Saved **{uploaded.name}**")
    csv_files = list_csv_files(DATA_ROOT, dataset)  # refresh

if not csv_files:
    st.sidebar.warning(f"No CSV files found in `{DATA_ROOT}/{dataset}/`. Upload one above or add files manually.")
    st.stop()

csv_file = st.sidebar.selectbox("CSV File", csv_files)

# 3. Attack type
attack_type = st.sidebar.selectbox("Attack Type", ["fgsm", "pgd", "both"])

# 4. XAI method
xai_method = st.sidebar.selectbox("XAI Method", ["both", "ig", "shap"])

# 5. Sample size
max_eval = st.sidebar.selectbox("Evaluation Sample Size", [500, 1000, 3000])

# 6. Run mode
st.sidebar.divider()
run_mode = st.sidebar.radio("Run Mode", ["Single Run", "Epsilon Sweep"], horizontal=True)

if run_mode == "Epsilon Sweep":
    st.sidebar.markdown("**Configure ε values** (up to 10):")
    n_eps = st.sidebar.slider("Number of ε values", min_value=2, max_value=10, value=4)
    default_eps = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    eps_list = []
    cols_eps = st.sidebar.columns(2)
    for i in range(n_eps):
        with cols_eps[i % 2]:
            val = st.number_input(f"ε{i+1}", value=default_eps[i] if i < len(default_eps) else 0.01,
                                  format="%.4f", key=f"eps_{i}")
            eps_list.append(val)
    eps_list = sorted(set(eps_list))

# 7. Hyperparameters (collapsible)
with st.sidebar.expander("Attack Hyperparameters"):
    epsilon = st.number_input("Epsilon (ε)", value=float(cfg["attack"]["epsilon"]), format="%.4f")
    alpha = st.number_input("Alpha (PGD step size)", value=float(cfg["attack"]["alpha"]), format="%.4f")
    pgd_iters = st.number_input("PGD Iterations", value=int(cfg["attack"]["iters"]), step=1)

with st.sidebar.expander("Model Hyperparameters"):
    epochs = st.number_input("Epochs", value=int(cfg["model"]["epochs"]), step=1)
    batch_size = st.number_input("Batch Size", value=int(cfg["model"]["batch_size"]), step=64)
    learning_rate = st.number_input("Learning Rate", value=float(cfg["model"]["lr"]), format="%.5f")

pre_run_errors, pre_run_warnings, pre_run_infos = build_pre_run_alerts(
    run_mode=run_mode,
    attack_type=attack_type,
    xai_method=xai_method,
    max_eval=max_eval,
    epsilon=epsilon,
    alpha=alpha,
    pgd_iters=int(pgd_iters),
    eps_list=eps_list if run_mode == "Epsilon Sweep" else None,
    cuda_requested=bool(cfg["run"].get("use_cuda", False)),
)

if pre_run_errors:
    for msg in pre_run_errors:
        st.sidebar.error(msg)
if pre_run_warnings:
    with st.sidebar.expander("Graph/Result Quality Warnings", expanded=True):
        for msg in pre_run_warnings:
            st.warning(msg)
        for msg in pre_run_infos[:2]:
            st.info(msg)

# ──────────────────────────────────────
# Run button
# ──────────────────────────────────────
run_button = st.sidebar.button(
    "🚀 Run Pipeline",
    type="primary",
    use_container_width=True,
    disabled=bool(pre_run_errors),
)

# ──────────────────────────────────────
# Main area — show instructions or results
# ──────────────────────────────────────
if not run_button:
    st.info("👈 Configure your experiment in the sidebar and click **Run Pipeline** to start.")

    # ── Introduction ──
    st.subheader("📖 What Is This?")
    st.markdown("""
This tool is part of a **Master's dissertation** by **Karim Khidr** for the
**MSc in Cybersecurity, Privacy and Trust** at **SETU** (South East Technological University).

The research investigates how **Explainable AI (XAI)** methods can detect **adversarial attacks**
against **Intrusion Detection Systems (IDS)**.

**The core idea:** When an attacker crafts adversarial network traffic to fool an IDS classifier,
the *feature attributions* (which features the model relies on) shift compared to clean inputs.
By measuring this **attribution drift**, we can detect adversarial manipulation even when the
model's final prediction doesn't change.

The pipeline works in three stages:
1. **Train** a neural network IDS on real network traffic data
2. **Attack** the trained model using gradient-based adversarial methods (FGSM / PGD)
3. **Explain** both clean and adversarial inputs using XAI (Integrated Gradients / SHAP),
   then measure how much the explanations drift — and whether that drift can reliably flag attacks
""")

    # ── How to Use ──
    st.subheader("🧭 How to Navigate")
    st.markdown("""
1. **Download a dataset** (see links below) and unzip the CSV file(s)
2. **Select the dataset** in the sidebar (`cicids2018` or `beth`)
3. **Upload a CSV file** using the file uploader in the sidebar, or place files manually
   in the `data/cicids2018/` or `data/beth/` folder
4. **Choose your experiment settings:**
   - **Attack Type** — FGSM (fast, single-step), PGD (iterative, stronger), or Both
   - **XAI Method** — Integrated Gradients, SHAP, or Both
   - **Sample Size** — 500 (quick test), 1000, or 3000 (more robust, slower)
   - Optionally tune ε (perturbation budget), epochs, batch size, and learning rate
5. **Click 🚀 Run Pipeline** and watch the live progress
6. **Review results** — timing breakdown, training curves, AUC metrics, ROC plots, drift histograms
   — each with dynamic interpretations that adapt to your specific configuration and results
7. **Download** all outputs as a ZIP for your thesis
""")

    # ── Datasets ──
    st.subheader("📂 Dataset Downloads")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
**🔹 CICIDS 2018**

A widely used benchmark for network intrusion detection containing labelled
network flows for various attack types (DoS, DDoS, Brute Force, Infiltration, etc.).

- 📥 [**Download from AWS Registry**](https://www.unb.ca/cic/datasets/ids-2018.html)
  — visit the page and follow the AWS download links
- Each attack type is a separate CSV (e.g., `DoS attacks-GoldenEye.csv`)
- Upload whichever CSV corresponds to the attack type you want to study
- The `Label` column is used for binary classification (Benign vs Attack)
""")
    with col_d2:
        st.markdown("""
**🔹 BETH Dataset**

Real-world host-based intrusion detection data from honeypot systems,
with `sus` (suspicious) and `evil` labels for binary classification.

- 📥 [**Download from Kaggle**](https://www.kaggle.com/datasets/katehighnam/beth-dataset)
  — requires a free Kaggle account
- Contains pre-split CSV files with network connection logs
- Upload the CSV file you want to analyse
- Labels are derived from `sus` and `evil` columns
""")

    st.divider()

    # ── Pipeline Details ──
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚙️ Pipeline Stages")
        st.markdown("""
| Stage | Description |
|-------|-------------|
| **1. Data Loading** | Load CSV, detect labels, handle NaN/Inf, scale features, stratified split |
| **2. Model Training** | MLP classifier with early stopping, LR scheduler, validation split |
| **3. Eval Subset** | Balanced selection of correctly-classified benign + attack samples |
| **4. Adversarial Attack** | FGSM (single gradient step) and/or PGD (iterative projected gradient) |
| **5. XAI Attribution** | Integrated Gradients and/or SHAP DeepExplainer on clean & adversarial inputs |
| **6. Drift Measurement** | Cosine similarity and Euclidean distance between attribution pairs |
| **7. ROC Evaluation** | Clean-pair attribution drift vs adversarial drift, with AUC and quality checks |
""")
    with col2:
        st.subheader("💡 Tips")
        st.markdown(f"""
- **First run?** Start with **FGSM + IG + 500 samples** — it's the fastest combination
- **For thesis figures**, use **Both attacks + Both XAI + 3000 samples** for comprehensive results
- **Epsilon (ε)** controls attack strength: small values (0.01) = subtle, large (0.3) = aggressive
- **PGD iterations** increase attack quality but also computation time
- **Compare runs** with different ε values to show how adversarial drift separates from clean-pair variation
- All results include **dynamic interpretations** that reference your specific settings

**Data folder structure:**
```
{DATA_ROOT}/
├── cicids2018/
│   ├── your-uploaded-file.csv
│   └── ...
└── beth/
    └── your-uploaded-file.csv
```
""")
    st.stop()

# ──────────────────────────────────────
# Pipeline Execution
# ──────────────────────────────────────

# Update cfg with sidebar values
cfg["data"]["dataset"] = dataset
cfg["run"]["max_eval"] = max_eval
cfg["attack"]["type"] = attack_type
cfg["attack"]["epsilon"] = epsilon
cfg["attack"]["alpha"] = alpha
cfg["attack"]["iters"] = pgd_iters
cfg["explain"]["method"] = xai_method
cfg["model"]["epochs"] = epochs
cfg["model"]["batch_size"] = batch_size
cfg["model"]["lr"] = learning_rate

# Status and progress containers
status_container = st.empty()
progress_bar = st.progress(0, text="Initialising...")
train_container = st.empty()

# Estimate total stages for progress bar
if run_mode == "Epsilon Sweep":
    n_atk = len([a for a in ["fgsm", "pgd"] if attack_type in (a, "both")])
    n_xai = len([x for x in ["ig", "shap"] if xai_method in (x, "both")])
    stage_count = {"current": 0, "total": 3 + n_xai + len(eps_list) * n_atk * (1 + n_xai) + 1}
else:
    stage_count = {"current": 0, "total": 12}

def status_cb(msg):
    stage_count["current"] += 1
    pct = min(stage_count["current"] / stage_count["total"], 1.0)
    progress_bar.progress(pct, text=msg)

def train_cb(epoch, total, train_loss, val_loss):
    train_container.text(f"  Training — Epoch {epoch}/{total}  |  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

try:
    if run_mode == "Epsilon Sweep":
        results = run_epsilon_sweep(
            cfg,
            csv_filename=csv_file,
            attack_type=attack_type,
            xai_method=xai_method,
            eps_list=eps_list,
            max_eval=max_eval,
            status_cb=status_cb,
            train_cb=train_cb,
        )
    else:
        results = run_pipeline(
            cfg,
            csv_filename=csv_file,
            attack_type=attack_type,
            xai_method=xai_method,
            max_eval=max_eval,
            status_cb=status_cb,
            train_cb=train_cb,
        )
except Exception as e:
    st.error(f"Pipeline failed: {e}")
    st.exception(e)
    st.stop()

progress_bar.progress(1.0, text="✅ Pipeline complete!")
train_container.empty()

# ══════════════════════════════════════
# EPSILON SWEEP RESULTS
# ══════════════════════════════════════
if run_mode == "Epsilon Sweep":
    st.divider()
    st.header("📊 Epsilon Sweep Results")

    st.subheader("🧪 Sweep Configuration")
    st.caption(
        f"**Dataset**: {dataset.upper()} — **File**: {csv_file} — "
        f"**Attack**: {attack_type.upper()} — **XAI**: {xai_method.upper()} — "
        f"**Samples**: {max_eval} — "
        f"**ε values**: {', '.join(str(e) for e in eps_list)}"
    )

    sweep_df = pd.DataFrame(results["sweep_results"])
    run_meta = results.get("run_metadata", {})
    if run_meta:
        model_meta = run_meta.get("model_metrics", {})
        data_meta = run_meta.get("data_metadata", {})
        eval_meta = run_meta.get("evaluation_subset", {})
        test_accuracy = model_meta.get("test_accuracy")
        test_accuracy_text = "?" if test_accuracy is None else f"{test_accuracy:.4f}"
        st.caption(
            f"Run provenance: **{data_meta.get('feature_count', '?')} features**, "
            f"test accuracy **{test_accuracy_text}**, "
            f"evaluation subset **{eval_meta.get('actual_total', '?')} / {eval_meta.get('requested_total', '?')}** samples, "
            f"device **{run_meta.get('device', '?')}**. Full metadata is saved in `run_metadata.json`."
        )

    result_warnings, result_infos = build_result_quality_alerts(results)
    show_alerts("⚠️ Result Quality Checks", warnings=result_warnings, infos=result_infos)

    # --- Timing ---
    st.subheader("⏱️ Timing Breakdown")
    timing_df = pd.DataFrame(results["timing"])
    timing_df.columns = ["Stage", "Time (s)"]
    total_rows = timing_df[timing_df["Stage"] == "TOTAL"]
    total_time = float(total_rows["Time (s)"].iloc[0]) if len(total_rows) else float(timing_df["Time (s)"].sum())
    st.caption(
        f"Total sweep time: **{total_time:.1f}s** across **{len(eps_list)} ε values**. "
        f"The model was trained **once** and reused for all ε values — "
        f"only the attack generation and XAI attribution stages are repeated per ε. "
        f"This is significantly faster than running {len(eps_list)} independent experiments."
    )
    st.dataframe(timing_df, width="stretch", hide_index=True)

    # --- Training History ---
    st.subheader("📈 Training History")
    hist_df = pd.DataFrame(results["history"])
    n_epochs_actual = len(hist_df)
    final_train = hist_df["train_loss"].iloc[-1]
    final_val = hist_df["val_loss"].iloc[-1]
    st.caption(
        f"Model trained for **{n_epochs_actual} epochs** (shared across all ε values). "
        f"Final loss — train: {final_train:.4f}, val: {final_val:.4f}."
    )
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig_tl, ax_tl = plt.subplots(figsize=(8, 5))
        ax_tl.plot(hist_df["epoch"], hist_df["train_loss"], linewidth=2.5, label="Train Loss")
        ax_tl.plot(hist_df["epoch"], hist_df["val_loss"], linewidth=2.5, label="Val Loss")
        ax_tl.set_xlabel("Epoch")
        ax_tl.set_ylabel("Loss")
        ax_tl.set_title("Training & Validation Loss")
        ax_tl.legend()
        ax_tl.grid(True, alpha=0.3)
        style_axis(ax_tl)
        fig_tl.tight_layout()
        st.pyplot(fig_tl); plt.close(fig_tl)

    # --- Summary Table ---
    st.subheader("📋 Sweep Summary Table")
    display_cols = ["epsilon", "attack", "xai", "preserved_count", "flip_count",
                    "mean_cosine_drift", "mean_clean_cosine_drift",
                    "mean_euclidean_drift", "mean_clean_euclidean_drift",
                    "auc_cosine", "auc_euclidean"]
    avail_cols = [c for c in display_cols if c in sweep_df.columns]
    st.caption(
        "Each row shows the results for one ε × attack × XAI combination. "
        "AUC compares adversarial attribution drift against natural clean-pair attribution drift. "
        "Higher ε often increases attack strength and flip rate, but AUC may stay low when perturbation drift "
        "is smaller than normal clean-sample explanation variation."
    )
    st.dataframe(sweep_df[avail_cols].round(4), width="stretch", hide_index=True)

    # --- Drift vs Epsilon Plots ---
    st.subheader("📈 Drift vs Epsilon")

    # Group by attack × xai for multi-line plots
    groups = sweep_df.groupby(["attack", "xai"])

    # Mean Cosine Drift
    fig_cos, ax_cos = plt.subplots(figsize=(10, 6))
    for (atk, xai), grp in groups:
        grp_sorted = grp.sort_values("epsilon")
        ax_cos.plot(grp_sorted["epsilon"], grp_sorted["mean_cosine_drift"],
                    marker="o", linewidth=2.5, markersize=7, label=f"{xai.upper()} + {atk.upper()}")
    ax_cos.set_xlabel("Epsilon (ε)")
    ax_cos.set_ylabel("Mean Cosine Drift")
    ax_cos.set_title("Epsilon Sweep: Mean Cosine Attribution Drift")
    ax_cos.legend()
    ax_cos.grid(True, alpha=0.3)
    style_axis(ax_cos)
    fig_cos.tight_layout()

    # Mean Euclidean Drift
    fig_euc, ax_euc = plt.subplots(figsize=(10, 6))
    for (atk, xai), grp in groups:
        grp_sorted = grp.sort_values("epsilon")
        ax_euc.plot(grp_sorted["epsilon"], grp_sorted["mean_euclidean_drift"],
                    marker="s", linewidth=2.5, markersize=7, label=f"{xai.upper()} + {atk.upper()}")
    ax_euc.set_xlabel("Epsilon (ε)")
    ax_euc.set_ylabel("Mean Euclidean Drift")
    ax_euc.set_title("Epsilon Sweep: Mean Euclidean Attribution Drift")
    ax_euc.legend()
    ax_euc.grid(True, alpha=0.3)
    style_axis(ax_euc)
    fig_euc.tight_layout()

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.pyplot(fig_cos); plt.close(fig_cos)
    with col_d2:
        st.pyplot(fig_euc); plt.close(fig_euc)

    st.caption(
        "These plots show adversarial attribution drift as ε changes. Interpret them alongside the clean-drift "
        "columns in the table: a larger adversarial drift is useful only if it separates from natural clean-pair "
        "attribution variation."
    )

    # --- AUC vs Epsilon Plot ---
    st.subheader("📉 Detection Performance (AUC) vs Epsilon")

    fig_auc, axes_auc = plt.subplots(1, 2, figsize=(14, 6))
    metric_labels = [("auc_cosine", "Cosine"), ("auc_euclidean", "Euclidean")]

    for ax, (col, name) in zip(axes_auc, metric_labels):
        for (atk, xai), grp in groups:
            grp_sorted = grp.sort_values("epsilon")
            ax.plot(grp_sorted["epsilon"], grp_sorted[col],
                    marker="o", linewidth=2.5, markersize=7, label=f"{xai.upper()} + {atk.upper()}")
        ax.set_xlabel("Epsilon (ε)")
        ax.set_ylabel("ROC-AUC")
        ax.set_title(f"{name} Drift Detection AUC")
        ax.set_ylim(0.0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
        ax.legend()
        ax.grid(True, alpha=0.3)
        style_axis(ax)
    fig_auc.tight_layout()
    st.pyplot(fig_auc); plt.close(fig_auc)

    # Dynamic caption for AUC plot
    valid_rows = sweep_df.dropna(subset=["auc_cosine"])
    if len(valid_rows) > 0:
        best_row = valid_rows.loc[valid_rows[["auc_cosine", "auc_euclidean"]].max(axis=1).idxmax()]
        worst_row = valid_rows.loc[valid_rows[["auc_cosine", "auc_euclidean"]].max(axis=1).idxmin()]
        best_auc = max(best_row["auc_cosine"], best_row["auc_euclidean"])
        worst_auc = max(worst_row["auc_cosine"], worst_row["auc_euclidean"])

        auc_caption = (
            f"**Best detection**: {best_row['xai'].upper()} + {best_row['attack'].upper()} "
            f"at ε={best_row['epsilon']} (best AUC = {best_auc:.4f}). "
            f"**Weakest**: {worst_row['xai'].upper()} + {worst_row['attack'].upper()} "
            f"at ε={worst_row['epsilon']} (best AUC = {worst_auc:.4f}). "
        )

        min_eps = valid_rows["epsilon"].min()
        max_eps = valid_rows["epsilon"].max()
        low_eps_auc = valid_rows[valid_rows["epsilon"] == min_eps][["auc_cosine", "auc_euclidean"]].max().max()
        high_eps_auc = valid_rows[valid_rows["epsilon"] == max_eps][["auc_cosine", "auc_euclidean"]].max().max()

        if high_eps_auc > low_eps_auc + 0.1:
            auc_caption += (
                f"Strict-baseline separation improves from ε={min_eps} (AUC={low_eps_auc:.4f}) "
                f"to ε={max_eps} (AUC={high_eps_auc:.4f}). This suggests larger perturbations move attributions "
                f"farther away from natural clean-pair variation, though they may also flip more predictions."
            )
        elif high_eps_auc < low_eps_auc + 0.02:
            auc_caption += (
                f"Strict-baseline AUC is relatively stable across ε values. This can mean adversarial drift remains "
                f"within the range of natural clean-sample explanation variation."
            )

        st.caption(auc_caption)

    # --- Flip Rate Plot ---
    st.subheader("🎯 Attack Success Rate vs Epsilon")
    fig_flip, ax_flip = plt.subplots(figsize=(10, 6))
    for (atk, xai), grp in groups:
        grp_sorted = grp.sort_values("epsilon")
        flip_rate = grp_sorted["flip_count"] / (grp_sorted["preserved_count"] + grp_sorted["flip_count"]) * 100
        ax_flip.plot(grp_sorted["epsilon"], flip_rate,
                     marker="o", linewidth=2.5, markersize=7, label=f"{xai.upper()} + {atk.upper()}")
    ax_flip.set_xlabel("Epsilon (ε)")
    ax_flip.set_ylabel("Flip Rate (%)")
    ax_flip.set_title("Attack Success Rate vs Epsilon")
    ax_flip.legend()
    ax_flip.grid(True, alpha=0.3)
    style_axis(ax_flip)
    fig_flip.tight_layout()
    st.pyplot(fig_flip); plt.close(fig_flip)
    st.caption(
        "The flip rate shows what percentage of predictions changed class after the attack. "
        "Higher ε typically increases the flip rate — but for drift-based detection, "
        "we focus on **preserved predictions** (where the model output is unchanged). "
        "A high flip rate with high AUC means the attack is both powerful and detectable."
    )

    # --- Download ---
    st.subheader("📥 Download Sweep Results")
    out_dir = results["out_dir"]
    st.markdown(f"All outputs saved to: `{out_dir}`")
    if results.get("summary_path"):
        st.caption(
            f"Thesis-ready summary table saved as `{os.path.basename(results['summary_path'])}`. "
            "Use this CSV for dissertation tables, comparison plots, and cross-run analysis."
        )

    import zipfile, io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for fname in os.listdir(out_dir):
            zf.write(os.path.join(out_dir, fname), fname)
    buf.seek(0)
    st.download_button(
        label="Download all sweep results (ZIP)",
        data=buf, file_name="xai_drift_sweep_results.zip", mime="application/zip",
    )

    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align:center; color:grey; font-size:0.85em;'>"
        "© 2025 Karim Khidr · MSc Cybersecurity, Privacy and Trust · "
        "<a href='https://www.setu.ie/' target='_blank'>SETU</a> "
        "(South East Technological University) · Dissertation Research Project"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ══════════════════════════════════════
# SINGLE RUN RESULTS (existing code below)
# ══════════════════════════════════════

# ──────────────────────────────────────
# Results Display
# ──────────────────────────────────────
st.divider()
st.header("📊 Results")

# ─── Configuration Recap ───
st.subheader("🧪 Experiment Configuration")
st.caption(
    f"**Dataset**: {dataset.upper()} — **File**: {csv_file} — "
    f"**Attack**: {attack_type.upper()} (ε={epsilon}"
    + (f", α={alpha}, {int(pgd_iters)} iters" if attack_type in ("pgd", "both") else "")
    + f") — **XAI**: {xai_method.upper()} — "
    f"**Samples**: {max_eval} — **Epochs**: {int(epochs)} — **Batch**: {int(batch_size)} — **LR**: {learning_rate}"
)

# ──────────────────────────────────────
# --- Timing Table ---
# ──────────────────────────────────────
st.subheader("⏱️ Timing Breakdown")
timing_df = pd.DataFrame(results["timing"])
timing_df.columns = ["Stage", "Time (s)"]
timing_no_total = timing_df[timing_df["Stage"] != "TOTAL"]
total_rows = timing_df[timing_df["Stage"] == "TOTAL"]
total_time = float(total_rows["Time (s)"].iloc[0]) if len(total_rows) else float(timing_no_total["Time (s)"].sum())
slowest = timing_no_total.loc[timing_no_total["Time (s)"].idxmax()]
slowest_name = slowest["Stage"]
slowest_pct = slowest["Time (s)"] / total_time * 100

# Identify time spent on each category
train_time = timing_df.loc[timing_df["Stage"].str.contains("Training", case=False), "Time (s)"].sum()
attack_time = timing_df.loc[timing_df["Stage"].str.contains("Attack", case=False), "Time (s)"].sum()
xai_time = timing_df.loc[timing_df["Stage"].str.contains("XAI", case=False), "Time (s)"].sum()
drift_roc_time = timing_df.loc[timing_df["Stage"].str.contains("Drift|ROC|Plots", case=False), "Time (s)"].sum()

# Compute throughput
samples_per_sec = max_eval / max(total_time, 0.01)

# Build timing interpretation
timing_notes = [
    f"Total pipeline time: **{total_time:.1f}s** for **{max_eval} samples**"
    f" ({samples_per_sec:.1f} samples/sec effective throughput). "
]

# Training commentary
if train_time > 0:
    train_pct = train_time / total_time * 100
    timing_notes.append(
        f"Model training took **{train_time:.1f}s** ({train_pct:.0f}%) over {int(epochs)} max epochs "
        f"with batch size {int(batch_size)}. "
    )
    if train_pct > 50:
        timing_notes.append(
            "Training dominates the pipeline — on resource-constrained hardware, "
            "consider reducing epochs or increasing batch size to trade a small accuracy drop for faster iteration. "
        )

# Attack commentary
if attack_time > 0:
    atk_pct = attack_time / total_time * 100
    timing_notes.append(
        f"Adversarial attack generation took **{attack_time:.1f}s** ({atk_pct:.0f}%). "
    )
    if attack_type == "pgd" or attack_type == "both":
        timing_notes.append(
            f"PGD is iterative ({int(pgd_iters)} steps per sample), making it significantly slower than "
            f"single-step FGSM. "
        )
    if attack_type == "both":
        fgsm_t = timing_df.loc[timing_df["Stage"].str.contains("FGSM", case=False), "Time (s)"].sum()
        pgd_t = timing_df.loc[timing_df["Stage"].str.contains("PGD", case=False), "Time (s)"].sum()
        if fgsm_t > 0 and pgd_t > 0:
            ratio = pgd_t / fgsm_t
            timing_notes.append(
                f"PGD was **{ratio:.1f}×** slower than FGSM ({pgd_t:.1f}s vs {fgsm_t:.1f}s), "
                f"reflecting the cost of {int(pgd_iters)} iterative gradient steps per sample. "
            )

# XAI commentary
if xai_time > 0:
    xai_pct = xai_time / total_time * 100
    timing_notes.append(
        f"XAI attribution took **{xai_time:.1f}s** ({xai_pct:.0f}%). "
    )
    if xai_method == "both":
        ig_t = timing_df.loc[timing_df["Stage"].str.contains("ig", case=False), "Time (s)"].sum()
        shap_t = timing_df.loc[timing_df["Stage"].str.contains("shap", case=False), "Time (s)"].sum()
        if ig_t > 0 and shap_t > 0:
            timing_notes.append(
                f"SHAP ({shap_t:.1f}s) vs IG ({ig_t:.1f}s) — "
            )
            if shap_t > ig_t * 1.5:
                timing_notes.append(
                    "SHAP is considerably more expensive because it computes Shapley values "
                    "using a background dataset, while IG uses a single interpolation path. "
                )
            else:
                timing_notes.append("Both methods had comparable cost for this sample size. ")
    elif xai_method == "shap":
        timing_notes.append(
            "SHAP (DeepExplainer) computes feature attributions using a background reference set, "
            "which is computationally heavier than gradient-based methods like IG. "
        )

# Scalability warning
if max_eval >= 1000:
    timing_notes.append(
        f"\n\n⚠️ **Scalability note**: With {max_eval} samples, "
        f"the pipeline processed {samples_per_sec:.1f} samples/sec. "
    )
    if total_time > 60:
        timing_notes.append(
            "Runtimes above 1 minute highlight a hardware limitation — "
            "larger sample sizes or more complex configurations (PGD + SHAP + both attacks) "
            "can place significant pressure on CPU/memory. "
            "A GPU-equipped machine would substantially reduce training and XAI computation times. "
        )
    elif total_time > 30:
        timing_notes.append(
            "This is manageable but would scale linearly with sample size. "
            "Running 3000 samples with both attacks and both XAI methods could take several minutes on this hardware. "
        )
else:
    timing_notes.append(
        f"\n\n💡 **Tip**: This run used {max_eval} samples. "
        f"Scaling to 3000 samples would take approximately **{total_time * 3000 / max_eval:.0f}s** "
        f"assuming linear scaling, though XAI stages may grow super-linearly with sample count."
    )

st.caption("".join(timing_notes))
st.dataframe(timing_df, width="stretch", hide_index=True)

run_meta = results.get("run_metadata", {})
if run_meta:
    model_meta = run_meta.get("model_metrics", {})
    data_meta = run_meta.get("data_metadata", {})
    eval_meta = run_meta.get("evaluation_subset", {})
    test_accuracy = model_meta.get("test_accuracy")
    test_accuracy_text = "?" if test_accuracy is None else f"{test_accuracy:.4f}"
    st.caption(
        f"Run provenance: **{data_meta.get('feature_count', '?')} features**, "
        f"test accuracy **{test_accuracy_text}**, "
        f"evaluation subset **{eval_meta.get('actual_total', '?')} / {eval_meta.get('requested_total', '?')}** samples, "
        f"device **{run_meta.get('device', '?')}**. Full metadata is saved in `run_metadata.json`."
    )

result_warnings, result_infos = build_result_quality_alerts(results)
show_alerts("⚠️ Result Quality Checks", warnings=result_warnings, infos=result_infos)

# ──────────────────────────────────────
# --- Training History ---
# ──────────────────────────────────────
st.subheader("📈 Training History")
hist_df = pd.DataFrame(results["history"])
n_epochs_actual = len(hist_df)
final_train = hist_df["train_loss"].iloc[-1]
final_val = hist_df["val_loss"].iloc[-1]
best_val_epoch = int(hist_df["val_loss"].idxmin()) + 1
best_val = hist_df["val_loss"].min()
gap = final_val - final_train

train_notes = []
if n_epochs_actual < int(epochs):
    train_notes.append(
        f"Early stopping triggered after **{n_epochs_actual}/{int(epochs)} epochs** "
        f"(best validation loss **{best_val:.4f}** at epoch {best_val_epoch}). "
        f"The model stopped training {int(epochs) - n_epochs_actual} epochs early to prevent over-fitting, "
        f"saving approximately **{train_time * (int(epochs) - n_epochs_actual) / max(n_epochs_actual, 1):.1f}s** "
        f"of unnecessary computation. "
    )
else:
    train_notes.append(
        f"Trained for all **{n_epochs_actual} epochs**. "
        f"Final training loss: {final_train:.4f}, validation loss: {final_val:.4f}. "
    )

if gap > 0.1:
    train_notes.append(
        f"The gap between training and validation loss ({gap:.4f}) indicates **over-fitting** — "
        f"the model memorises training patterns that don't generalise. "
        f"Consider reducing epochs, increasing dropout, or using a larger dataset. "
    )
elif gap > 0.05:
    train_notes.append(
        f"Slight over-fitting detected (gap = {gap:.4f}). The model still generalises reasonably well, "
        f"but further tuning (e.g., fewer epochs or stronger regularisation) could improve robustness. "
    )
else:
    train_notes.append(
        f"Training and validation losses are closely aligned (gap = {gap:.4f}), indicating "
        f"**good generalisation** — the model learns patterns that transfer to unseen data. "
    )

# Relate to sample size
train_notes.append(
    f"\n\nThe model was trained on the full training split of **{csv_file}**, "
    f"then evaluated on a balanced subset of **{max_eval}** samples. "
    f"Using a larger evaluation sample (e.g., 3000) provides more statistically robust AUC estimates "
    f"but increases XAI computation time significantly."
)
st.caption("".join(train_notes))

col_t1, col_t2 = st.columns(2)
with col_t1:
    fig_tl, ax_tl = plt.subplots(figsize=(8, 5))
    ax_tl.plot(hist_df["epoch"], hist_df["train_loss"], linewidth=2.5, label="Train Loss")
    ax_tl.plot(hist_df["epoch"], hist_df["val_loss"], linewidth=2.5, label="Val Loss")
    ax_tl.set_xlabel("Epoch")
    ax_tl.set_ylabel("Loss")
    ax_tl.set_title("Training & Validation Loss")
    ax_tl.legend()
    ax_tl.grid(True, alpha=0.3)
    style_axis(ax_tl)
    fig_tl.tight_layout()
    st.pyplot(fig_tl)
    plt.close(fig_tl)

# ──────────────────────────────────────
# --- Metrics Summary ---
# ──────────────────────────────────────
st.subheader("📋 Metrics Summary")
metrics = results["metrics"]

# Build dynamic overview
all_aucs = []
for key, vals in metrics.items():
    for m in ("auc_cosine", "auc_euclidean"):
        if pd.notna(vals[m]):
            all_aucs.append((key, m.replace("auc_", ""), vals[m]))

if all_aucs:
    best = max(all_aucs, key=lambda x: x[2])
    worst = min(all_aucs, key=lambda x: x[2])
    mean_auc = sum(x[2] for x in all_aucs) / len(all_aucs)
else:
    best = ("n/a", "n/a", float("nan"))
    worst = ("n/a", "n/a", float("nan"))
    mean_auc = float("nan")

metrics_notes = [
    f"Across **{len(metrics)} configuration(s)** (ε={epsilon}), "
    f"the mean AUC is **{mean_auc:.4f}**. "
    f"AUC is measured against clean-pair attribution drift, so it tests whether adversarial drift separates from "
    f"natural clean-sample explanation variation. The **best separation** was **{best[0].upper()}** using "
    f"**{best[1]}** distance (AUC = {best[2]:.4f})"
]

if worst[2] < 0.6:
    metrics_notes.append(
        f". The **weakest** was **{worst[0].upper()}** with **{worst[1]}** (AUC = {worst[2]:.4f}) — "
        f"adversarial drift is smaller than or heavily overlapped with clean-pair attribution variation for that combination"
    )
elif worst[2] < 0.75:
    metrics_notes.append(
        f". The **weakest** was **{worst[0].upper()}** with **{worst[1]}** (AUC = {worst[2]:.4f}) — "
        f"strict-baseline separation is weak at this level and would need careful thresholding"
    )

metrics_notes.append(". ")

# Epsilon interpretation
if epsilon < 0.01:
    metrics_notes.append(
        f"With a small ε={epsilon}, perturbations are subtle and harder to detect — "
        f"low strict-baseline AUC scores are expected when the attack barely changes attributions compared with clean variation. "
    )
elif epsilon > 0.1:
    metrics_notes.append(
        f"With ε={epsilon} (a relatively large perturbation budget), "
        f"the attack has significant room to alter inputs. "
    )
    if mean_auc > 0.85:
        metrics_notes.append(
            "The high AUC values confirm that XAI attributions shift measurably under strong perturbations, "
            "making drift-based detection effective. "
        )
    elif mean_auc < 0.7:
        metrics_notes.append(
            "Despite the large ε, strict-baseline AUC is low — this indicates adversarial attribution shifts "
            "do not separate cleanly from natural clean-sample variation. "
        )
else:
    metrics_notes.append(
        f"ε={epsilon} is a moderate perturbation budget — "
        f"{'strict-baseline separation works well at this level' if mean_auc > 0.8 else 'the attack operates near or inside natural attribution variation'}. "
    )

st.caption("".join(metrics_notes))

for key, vals in metrics.items():
    total_samples = vals["n_preserved"] + vals["n_flipped"]
    flip_pct = vals["n_flipped"] / max(total_samples, 1) * 100
    best_metric = max(
        ("Cosine", vals["auc_cosine"]),
        ("Euclidean", vals["auc_euclidean"]),
        key=lambda x: x[1],
    )

    # Parse XAI and attack from the key (e.g., "ig_fgsm")
    parts = key.split("_")
    xai_name = parts[0].upper() if len(parts) > 0 else "?"
    atk_name = parts[1].upper() if len(parts) > 1 else "?"

    with st.expander(
        f"**{key.upper()}**  —  AUC Cos: {vals['auc_cosine']:.4f}  |  "
        f"AUC Euc: {vals['auc_euclidean']:.4f}"
    ):
        # Per-config dynamic caption
        config_notes = []
        config_notes.append(
            f"**{atk_name}** attack (ε={epsilon}) with **{xai_name}** attributions: "
        )
        if flip_pct > 50:
            config_notes.append(
                f"The attack flipped **{vals['n_flipped']}/{total_samples}** predictions "
                f"(**{flip_pct:.1f}% success rate**) — the majority of samples changed class, "
                f"indicating a strong attack at this ε. "
            )
        elif flip_pct > 20:
            config_notes.append(
                f"The attack flipped **{vals['n_flipped']}/{total_samples}** predictions "
                f"({flip_pct:.1f}%). A moderate success rate — the model is partially robust. "
            )
        else:
            config_notes.append(
                f"Only **{vals['n_flipped']}/{total_samples}** predictions flipped ({flip_pct:.1f}%) — "
                f"the model is fairly robust to this perturbation level. "
            )

        config_notes.append(
            f"Among the **{vals['n_preserved']}** preserved-prediction samples, "
            f"**{best_metric[0]}** distance had the best clean-pair baseline separation (AUC = {best_metric[1]:.4f}). "
        )

        if best_metric[1] > 0.9:
            config_notes.append(
                f"This is excellent — {xai_name} attributions shift strongly under {atk_name}, "
                f"and {best_metric[0]} distance separates that shift from natural clean-pair variation."
            )
        elif best_metric[1] > 0.75:
            config_notes.append(
                f"Separation is moderate — {xai_name} picks up some attribution drift from {atk_name}, "
                f"but there is overlap with clean-pair attribution variation."
            )
        else:
            config_notes.append(
                f"Strict-baseline separation is weak — {xai_name} attributions do not shift enough under {atk_name} "
                f"for {best_metric[0]} distance to exceed natural clean-sample variation."
            )

        st.caption("".join(config_notes))

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("AUC Cosine", f"{vals['auc_cosine']:.4f}")
        col_m2.metric("AUC Euclidean", f"{vals['auc_euclidean']:.4f}")

        col_m4, col_m5 = st.columns(2)
        col_m4.metric("Preserved Samples", vals["n_preserved"])
        col_m5.metric("Flipped (Attack Success)", vals["n_flipped"])

# ──────────────────────────────────────
# --- ROC Curves ---
# ──────────────────────────────────────
st.subheader("📉 ROC Curves")
figures = results["figures"]
roc_keys = [k for k in figures if "roc" in k]
n_roc = len(roc_keys)

roc_notes = [
    f"**{n_roc} ROC curve(s)** generated for {max_eval} evaluation samples at ε={epsilon}. "
    f"Each curve uses clean-pair attribution drift as the negative class and adversarial drift as the positive class, "
    f"then plots True Positive Rate against False Positive Rate. "
]
if best[2] > 0.9:
    roc_notes.append(
        "The best curves hug the top-left corner, confirming strong detection — "
        "these results are suitable for thesis figures demonstrating effective drift-based detection. "
    )
if attack_type == "both":
    roc_notes.append(
        "Compare FGSM vs PGD curves: PGD (iterative, multi-step) often produces more transferable "
        "perturbations that can be either easier or harder to detect depending on the XAI method. "
    )
if xai_method == "both":
    roc_notes.append(
        "Compare IG vs SHAP curves: differences reveal which attribution method is more sensitive "
        "to adversarial manipulation of the input features. "
    )
st.caption("".join(roc_notes))

cols = st.columns(min(3, len(roc_keys)))
for idx, key in enumerate(sorted(roc_keys)):
    with cols[idx % 3]:
        st.pyplot(figures[key])

# ──────────────────────────────────────
# --- Drift Histograms (from saved files) ---
# ──────────────────────────────────────
st.subheader("📊 Drift Histograms")
out_dir = results["out_dir"]
hist_files = sorted([f for f in os.listdir(out_dir) if f.startswith("hist_") and f.endswith(".png")])
n_hist = len(hist_files)
if n_hist > 0:
    hist_notes = [
        f"**{n_hist} histogram(s)** showing the distribution of drift distances for "
        f"clean vs adversarial samples (ε={epsilon}, {max_eval} samples). "
    ]
    if epsilon > 0.05:
        hist_notes.append(
            "With this perturbation level, well-separated distributions (distinct peaks) "
            "confirm that the XAI method captures meaningful attribution changes. "
        )
    else:
        hist_notes.append(
            "At this small ε, distributions may overlap considerably — "
            "this is expected since subtle perturbations cause minimal attribution shift. "
        )
    hist_notes.append(
        "Overlapping histograms represent a limitation: the drift metric cannot cleanly separate "
        "adversarial from legitimate traffic, which would need to be addressed in a production IDS. "
        "Well-separated histograms indicate the metric is a strong candidate for real-world deployment."
    )
    st.caption("".join(hist_notes))

cols_h = st.columns(min(3, max(len(hist_files), 1)))
for idx, fname in enumerate(hist_files):
    with cols_h[idx % 3]:
        st.image(os.path.join(out_dir, fname), caption=fname.replace("hist_", "").replace(".png", "").replace("_", " ").title())

# --- Download ---
st.subheader("📥 Download Results")
st.markdown(f"All outputs saved to: `{out_dir}`")
if results.get("summary_path"):
    st.caption(
        f"Thesis-ready summary table saved as `{os.path.basename(results['summary_path'])}`. "
        "Use this CSV for dissertation tables, comparison plots, and cross-run analysis."
    )

import zipfile, io

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as zf:
    for fname in os.listdir(out_dir):
        zf.write(os.path.join(out_dir, fname), fname)
buf.seek(0)

st.download_button(
    label="Download all results (ZIP)",
    data=buf,
    file_name="xai_drift_results.zip",
    mime="application/zip",
)

# ── Footer ──
st.divider()
st.markdown(
    "<div style='text-align:center; color:grey; font-size:0.85em;'>"
    "© 2025 Karim Khidr · MSc Cybersecurity, Privacy and Trust · "
    "<a href='https://www.setu.ie/' target='_blank'>SETU</a> "
    "(South East Technological University) · Dissertation Research Project"
    "</div>",
    unsafe_allow_html=True,
)
