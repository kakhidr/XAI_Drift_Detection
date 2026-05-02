"""
XAI Drift Detection — Streamlit Web Interface
Run with:  streamlit run app.py
"""
import os
import sys
import yaml
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from src.data.loader import list_csv_files
from src.pipeline.run_experiment import run_pipeline

# ──────────────────────────────────────
# Page config
# ──────────────────────────────────────
st.set_page_config(page_title="XAI Drift Detection", layout="wide")
st.title("🔍 XAI Drift Detection Pipeline")
st.markdown("Detect adversarial manipulation via attribution drift in intrusion detection systems.")

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

# 2. CSV file selection
csv_files = list_csv_files(DATA_ROOT, dataset)
if not csv_files:
    st.sidebar.warning(f"No CSV files found in `{DATA_ROOT}/{dataset}/`. Please add your data files.")
    st.stop()

csv_file = st.sidebar.selectbox("CSV File", csv_files)

# 3. Attack type
attack_type = st.sidebar.selectbox("Attack Type", ["fgsm", "pgd", "both"])

# 4. XAI method
xai_method = st.sidebar.selectbox("XAI Method", ["both", "ig", "shap"])

# 5. Sample size
max_eval = st.sidebar.selectbox("Evaluation Sample Size", [500, 1000, 3000])

# 6. Hyperparameters (collapsible)
with st.sidebar.expander("Attack Hyperparameters"):
    epsilon = st.number_input("Epsilon (ε)", value=float(cfg["attack"]["epsilon"]), format="%.4f")
    alpha = st.number_input("Alpha (PGD step size)", value=float(cfg["attack"]["alpha"]), format="%.4f")
    pgd_iters = st.number_input("PGD Iterations", value=int(cfg["attack"]["iters"]), step=1)

with st.sidebar.expander("Model Hyperparameters"):
    epochs = st.number_input("Epochs", value=int(cfg["model"]["epochs"]), step=1)
    batch_size = st.number_input("Batch Size", value=int(cfg["model"]["batch_size"]), step=64)
    learning_rate = st.number_input("Learning Rate", value=float(cfg["model"]["lr"]), format="%.5f")

# ──────────────────────────────────────
# Run button
# ──────────────────────────────────────
run_button = st.sidebar.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# ──────────────────────────────────────
# Main area — show instructions or results
# ──────────────────────────────────────
if not run_button:
    st.info("Configure your experiment in the sidebar and click **Run Pipeline** to start.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pipeline Stages")
        st.markdown("""
        1. **Data Loading** — Load & preprocess CSV, binary labels, scaling
        2. **Model Training** — MLP with early stopping + validation
        3. **Eval Subset** — Balanced correctly-classified samples
        4. **Adversarial Attack** — FGSM and/or PGD perturbations
        5. **XAI Attribution** — Integrated Gradients and/or SHAP
        6. **Drift Measurement** — Cosine, Euclidean, KL divergence
        7. **ROC Evaluation** — AUC curves for drift-based detection
        """)
    with col2:
        st.subheader("Data Setup")
        st.markdown(f"""
        Place your datasets under the `{DATA_ROOT}/` directory:
        ```
        {DATA_ROOT}/
        ├── cicids2018/
        │   ├── DoS attacks-GoldenEye.csv
        │   └── ...
        └── beth/
            └── beth.csv
        ```
        """)
    st.stop()

# ──────────────────────────────────────
# Pipeline Execution
# ──────────────────────────────────────

# Update cfg with sidebar values
cfg["data"]["dataset"] = dataset
cfg["attack"]["epsilon"] = epsilon
cfg["attack"]["alpha"] = alpha
cfg["attack"]["iters"] = pgd_iters
cfg["model"]["epochs"] = epochs
cfg["model"]["batch_size"] = batch_size
cfg["model"]["lr"] = learning_rate

# Status and progress containers
status_container = st.empty()
progress_bar = st.progress(0, text="Initialising...")
train_container = st.empty()

stage_count = {"current": 0, "total": 12}

def status_cb(msg):
    stage_count["current"] += 1
    pct = min(stage_count["current"] / stage_count["total"], 1.0)
    progress_bar.progress(pct, text=msg)

def train_cb(epoch, total, train_loss, val_loss):
    train_container.text(f"  Training — Epoch {epoch}/{total}  |  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

try:
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

# ──────────────────────────────────────
# Results Display
# ──────────────────────────────────────
st.divider()
st.header("📊 Results")

# --- Timing Table ---
st.subheader("⏱️ Timing Breakdown")
timing_df = pd.DataFrame(results["timing"])
timing_df.columns = ["Stage", "Time (s)"]
st.dataframe(timing_df, use_container_width=True, hide_index=True)

# --- Training History ---
st.subheader("📈 Training History")
hist_df = pd.DataFrame(results["history"])
col_t1, col_t2 = st.columns(2)
with col_t1:
    fig_tl, ax_tl = plt.subplots()
    ax_tl.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")
    ax_tl.plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss")
    ax_tl.set_xlabel("Epoch")
    ax_tl.set_ylabel("Loss")
    ax_tl.legend()
    ax_tl.set_title("Training & Validation Loss")
    st.pyplot(fig_tl)
    plt.close(fig_tl)

# --- Metrics Summary ---
st.subheader("📋 Metrics Summary")
metrics = results["metrics"]
for key, vals in metrics.items():
    with st.expander(f"**{key.upper()}**  —  AUC Cos: {vals['auc_cosine']:.4f}  |  AUC Euc: {vals['auc_euclidean']:.4f}  |  AUC KL: {vals['auc_kl']:.4f}"):
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("AUC Cosine", f"{vals['auc_cosine']:.4f}")
        col_m2.metric("AUC Euclidean", f"{vals['auc_euclidean']:.4f}")
        col_m3.metric("AUC KL", f"{vals['auc_kl']:.4f}")

        col_m4, col_m5 = st.columns(2)
        col_m4.metric("Preserved Samples", vals["n_preserved"])
        col_m5.metric("Flipped (Attack Success)", vals["n_flipped"])

# --- ROC Curves ---
st.subheader("📉 ROC Curves")
figures = results["figures"]
roc_keys = [k for k in figures if "roc" in k]

cols = st.columns(min(3, len(roc_keys)))
for idx, key in enumerate(sorted(roc_keys)):
    with cols[idx % 3]:
        st.pyplot(figures[key])

# --- Drift Histograms (from saved files) ---
st.subheader("📊 Drift Histograms")
out_dir = results["out_dir"]
hist_files = sorted([f for f in os.listdir(out_dir) if f.startswith("hist_") and f.endswith(".png")])

cols_h = st.columns(min(3, max(len(hist_files), 1)))
for idx, fname in enumerate(hist_files):
    with cols_h[idx % 3]:
        st.image(os.path.join(out_dir, fname), caption=fname.replace("hist_", "").replace(".png", "").replace("_", " ").title())

# --- Download ---
st.subheader("📥 Download Results")
st.markdown(f"All outputs saved to: `{out_dir}`")

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
