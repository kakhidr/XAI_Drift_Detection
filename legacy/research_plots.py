#!/usr/bin/env python3
"""
Dissertation Figure Generator
==============================
Generates publication-ready plots from XAI Drift Detection experiment results.

Lives in legacy/ to avoid modifying the main pipeline.

Usage
-----
  # Generate all plots from saved sweep CSV (no re-run needed):
  python legacy/dissertation_plots.py

  # Re-run pipeline to get raw drift scores, then generate all plots:
  python legacy/dissertation_plots.py --rerun

  # Custom paths:
  python legacy/dissertation_plots.py --sweep-csv results/logs/20260504_001452_sweep/epsilon_sweep_results.csv --output-dir legacy/figures

Author: Karim Khidr
"""
import argparse
import os
import sys
import tempfile

# ── matplotlib setup (must precede any other mpl import) ─────────────
_MPL_DIR = os.path.join(tempfile.gettempdir(), "xai_drift_matplotlib")
os.makedirs(_MPL_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Add project root to path so we can import src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.eval.plot_style import style_axis

# ── Style defaults ───────────────────────────────────────────────────
STYLE = {
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
}
plt.rcParams.update(STYLE)

COLOURS = {
    "clean": "#5B9BD5",   # blue
    "fgsm":  "#F4A261",   # orange
    "pgd":   "#E76F51",   # red-orange
    "ig":    "#2A9D8F",   # teal
    "shap":  "#E9C46A",   # gold
}
METRICS = ("cosine", "euclidean")
METRIC_LABELS = {"cosine": "Cosine Distance", "euclidean": "Euclidean Distance"}


def _save(fig, out_dir, name):
    """Save figure as both PNG and PDF."""
    for ax in fig.axes:
        style_axis(ax)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"  -> {name}.png / .pdf")


# ══════════════════════════════════════════════════════════════════════
# PLOT 1: Epsilon vs AUC (combined: Cosine + Euclidean side by side)
# ══════════════════════════════════════════════════════════════════════
def plot_epsilon_vs_auc(df, out_dir):
    """Line chart: detection AUC vs epsilon for both metrics in one figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, metric in enumerate(METRICS):
        ax = axes[i]
        col = f"auc_{metric}"
        for (xai, attack), grp in df.groupby(["xai", "attack"]):
            grp_s = grp.sort_values("epsilon")
            label = f"{xai.upper()} + {attack.upper()}"
            ax.plot(grp_s["epsilon"], grp_s[col], "o-", label=label, linewidth=2.5, markersize=7)
        ax.set_xlabel("Perturbation Strength (ε)")
        ax.set_ylabel("AUC" if i == 0 else "")
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylim(-0.05, 1.1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Detection AUC vs Perturbation Strength", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig_epsilon_vs_auc")


# ══════════════════════════════════════════════════════════════════════
# PLOT 2: Epsilon vs Mean Drift (log scale, combined)
# ══════════════════════════════════════════════════════════════════════
def plot_epsilon_vs_drift(df, out_dir):
    """Log-scale drift magnitude vs epsilon — Cosine & Euclidean combined."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, metric in enumerate(METRICS):
        ax = axes[i]
        col = f"mean_{metric}_drift"
        for (xai, attack), grp in df.groupby(["xai", "attack"]):
            grp_s = grp.sort_values("epsilon")
            vals = grp_s[col].abs()
            label = f"{xai.upper()} + {attack.upper()}"
            ax.plot(grp_s["epsilon"], vals, "s-", label=label, linewidth=2.5, markersize=7)
        ax.set_xlabel("Perturbation Strength (ε)")
        ax.set_ylabel(f"Mean {METRIC_LABELS[metric]}" if i == 0 else "")
        ax.set_title(METRIC_LABELS[metric])
        ax.set_yscale("log")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Mean Attribution Drift vs Perturbation Strength", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig_epsilon_vs_drift")


# ══════════════════════════════════════════════════════════════════════
# PLOT 3: IG vs SHAP grouped bar comparison (combined)
# ══════════════════════════════════════════════════════════════════════
def plot_xai_comparison_bars(df, out_dir):
    """Grouped bar chart: IG vs SHAP AUC — Cosine & Euclidean in one figure."""
    attack = "fgsm" if "fgsm" in df["attack"].values else df["attack"].iloc[0]
    sub = df[df["attack"] == attack].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, metric in enumerate(METRICS):
        ax = axes[i]
        col = f"auc_{metric}"
        pivot = sub.pivot(index="epsilon", columns="xai", values=col)
        pivot.plot(kind="bar", ax=ax, color=[COLOURS["ig"], COLOURS["shap"]], edgecolor="black", width=0.7)
        ax.set_xlabel("ε")
        ax.set_ylabel("AUC" if i == 0 else "")
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylim(0, 1.15)
        ax.legend(title="XAI Method", labels=["IG", "SHAP"])
        ax.set_xticklabels([f"{v:.3f}" for v in pivot.index], rotation=45)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"IG vs SHAP Detection Performance ({attack.upper()})", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig_xai_comparison_bars")


# ══════════════════════════════════════════════════════════════════════
# PLOT 4: AUC Heatmap (XAI × Metric × Epsilon)
# ══════════════════════════════════════════════════════════════════════
def plot_auc_heatmap(df, out_dir):
    """Compact heatmap summarising all AUC results."""
    attack = "fgsm" if "fgsm" in df["attack"].values else df["attack"].iloc[0]
    sub = df[df["attack"] == attack].copy()

    rows = []
    for _, r in sub.iterrows():
        for metric in METRICS:
            rows.append({
                "ε": r["epsilon"],
                "Config": f"{r['xai'].upper()} — {METRIC_LABELS[metric]}",
                "AUC": r[f"auc_{metric}"],
            })
    heat = pd.DataFrame(rows).pivot(index="Config", columns="ε", values="AUC")

    fig, ax = plt.subplots(figsize=(12, 5))
    values = heat.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)
    im = ax.imshow(masked_values, cmap="YlGnBu", vmin=0.4, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("AUC")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_xticklabels([f"{v:g}" for v in heat.columns])
    ax.set_yticklabels(heat.index)
    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            val = values[row, col]
            if np.isfinite(val):
                ax.text(col, row, f"{val:.3f}", ha="center", va="center", fontsize=13)
    ax.set_title(f"Detection AUC Summary ({attack.upper()})")
    ax.set_xlabel("Perturbation Strength (ε)")
    ax.set_ylabel("")
    fig.tight_layout()
    _save(fig, out_dir, "fig_auc_heatmap")


# ══════════════════════════════════════════════════════════════════════
# PLOT 5: Flip Rate vs Epsilon
# ══════════════════════════════════════════════════════════════════════
def plot_flip_rate(df, out_dir):
    """Shows how many predictions flip as epsilon increases."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for (xai, attack), grp in df.groupby(["xai", "attack"]):
        grp_s = grp.sort_values("epsilon")
        total = grp_s["preserved_count"] + grp_s["flip_count"]
        flip_rate = grp_s["flip_count"] / total
        ax.plot(grp_s["epsilon"], flip_rate, "o-", label=f"{xai.upper()} + {attack.upper()}",
                linewidth=2.5, markersize=7)

    ax.set_xlabel("Perturbation Strength (ε)")
    ax.set_ylabel("Prediction Flip Rate")
    ax.set_title("Adversarial Prediction Flip Rate vs ε")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fig.tight_layout()
    _save(fig, out_dir, "fig_flip_rate")


# ══════════════════════════════════════════════════════════════════════
# PLOT 6: Clean Explanation Stability vs Adversarial Drift
#          COMBINED 2×2 grid: (FGSM | PGD) × (Cosine | Euclidean)
# ══════════════════════════════════════════════════════════════════════
def plot_clean_vs_adversarial_combined(raw_scores, out_dir):
    """
    2×2 combined figure per XAI method:
      Rows: FGSM, PGD
      Cols: Cosine Distance, Euclidean Distance

    Matches the dissertation reference style with overlaid histograms.
    """
    xai_methods = sorted(set(k[0] for k in raw_scores.keys()))
    attacks = ["fgsm", "pgd"]

    for xai in xai_methods:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for row, atk in enumerate(attacks):
            if (xai, atk) not in raw_scores:
                for col in range(2):
                    axes[row, col].text(0.5, 0.5, "No data", ha="center", va="center")
                continue
            scores = raw_scores[(xai, atk)]

            for col, metric in enumerate(METRICS):
                ax = axes[row, col]
                clean = scores[f"clean_{metric}"]
                adv = scores[f"adv_{metric}"]

                # Shared bin edges up to 99.5th percentile
                all_vals = np.concatenate([clean, adv])
                bins = np.linspace(0, np.percentile(all_vals, 99.5), 40)

                ax.hist(clean, bins=bins, alpha=0.7, color=COLOURS["clean"],
                        edgecolor="white", label="Clean vs Clean")
                ax.hist(adv, bins=bins, alpha=0.7, color=COLOURS[atk],
                        edgecolor="white", label=atk.upper())

                ax.set_title(f"Clean Explanation Stability vs {atk.upper()} Drift")
                ax.set_xlabel(METRIC_LABELS[metric])
                ax.set_ylabel("Frequency")
                ax.legend(loc="upper right")

        fig.suptitle(f"Attribution Drift Distributions — {xai.upper()}", fontsize=18, fontweight="bold", y=1.01)
        fig.tight_layout()
        _save(fig, out_dir, f"fig_clean_vs_adv_combined_{xai}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 7: Individual Clean vs Adversarial (one per attack+metric)
#          for maximum flexibility in dissertation layout
# ══════════════════════════════════════════════════════════════════════
def plot_clean_vs_adversarial_individual(raw_scores, out_dir):
    """
    Individual histogram: Clean vs Adversarial for each (xai, attack, metric).
    Produces: fig_stability_ig_fgsm_cosine, fig_stability_ig_pgd_euclidean, etc.
    """
    xai_methods = sorted(set(k[0] for k in raw_scores.keys()))
    attacks = ["fgsm", "pgd"]

    for xai in xai_methods:
        for atk in attacks:
            if (xai, atk) not in raw_scores:
                continue
            scores = raw_scores[(xai, atk)]

            for metric in METRICS:
                clean = scores[f"clean_{metric}"]
                adv = scores[f"adv_{metric}"]

                all_vals = np.concatenate([clean, adv])
                bins = np.linspace(0, np.percentile(all_vals, 99.5), 40)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(clean, bins=bins, alpha=0.7, color=COLOURS["clean"],
                        edgecolor="white", label="Clean vs Clean")
                ax.hist(adv, bins=bins, alpha=0.7, color=COLOURS[atk],
                        edgecolor="white", label=atk.upper())

                ax.set_title(f"Clean Explanation Stability vs {atk.upper()} Drift ({xai.upper()})")
                ax.set_xlabel(METRIC_LABELS[metric])
                ax.set_ylabel("Frequency")
                ax.legend(loc="upper right")
                fig.tight_layout()
                _save(fig, out_dir, f"fig_stability_{xai}_{atk}_{metric}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 8: ROC Overlay — Cosine + Euclidean on one figure
# ══════════════════════════════════════════════════════════════════════
def plot_roc_overlay(raw_scores, out_dir):
    """Overlay ROC curves for Cosine and Euclidean on one figure per (xai, attack)."""
    from sklearn.metrics import roc_curve, auc

    for (xai, atk), scores in raw_scores.items():
        fig, ax = plt.subplots(figsize=(8.5, 8.5))

        colours = {"cosine": "#2A9D8F", "euclidean": "#E76F51"}
        for metric in METRICS:
            clean = scores[f"clean_{metric}"]
            adv = scores[f"adv_{metric}"]
            y_labels = np.concatenate([np.zeros(len(clean)), np.ones(len(adv))])
            all_scores = np.concatenate([clean, adv])
            fpr, tpr, _ = roc_curve(y_labels, all_scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colours[metric], lw=3,
                    label=f"{METRIC_LABELS[metric]} (AUC = {roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Comparison — {xai.upper()} + {atk.upper()}")
        ax.legend(loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        _save(fig, out_dir, f"fig_roc_overlay_{xai}_{atk}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 9: Combined ROC — All 4 configs (IG+FGSM, IG+PGD, SHAP+FGSM, SHAP+PGD)
#          in a 2×2 grid to save dissertation space
# ══════════════════════════════════════════════════════════════════════
def plot_roc_combined(raw_scores, out_dir):
    """2×2 ROC grid: rows=XAI method, cols=attack type."""
    from sklearn.metrics import roc_curve, auc

    xai_methods = sorted(set(k[0] for k in raw_scores.keys()))
    attacks = ["fgsm", "pgd"]
    colours = {"cosine": "#2A9D8F", "euclidean": "#E76F51"}

    fig, axes = plt.subplots(len(xai_methods), len(attacks), figsize=(15, 11))
    if len(xai_methods) == 1:
        axes = axes.reshape(1, -1)

    for row, xai in enumerate(xai_methods):
        for col, atk in enumerate(attacks):
            ax = axes[row, col]
            if (xai, atk) not in raw_scores:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            scores = raw_scores[(xai, atk)]
            for metric in METRICS:
                clean = scores[f"clean_{metric}"]
                adv = scores[f"adv_{metric}"]
                y_labels = np.concatenate([np.zeros(len(clean)), np.ones(len(adv))])
                all_s = np.concatenate([clean, adv])
                fpr, tpr, _ = roc_curve(y_labels, all_s)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colours[metric], lw=3,
                        label=f"{METRIC_LABELS[metric]} (AUC={roc_auc:.4f})")

            ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
            ax.set_title(f"{xai.upper()} + {atk.upper()}")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend(loc="lower right")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)

    fig.suptitle("ROC Curves — Drift-Based Adversarial Detection", fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "fig_roc_combined")


# ══════════════════════════════════════════════════════════════════════
# PLOT 10: Top-K Feature Attribution Shift (combined 2×2)
# ══════════════════════════════════════════════════════════════════════
def plot_topk_feature_shift(attribution_data, out_dir, k=15):
    """2×2 horizontal bar chart: rows=XAI, cols=attack."""
    xai_methods = sorted(set(key[0] for key in attribution_data.keys()))
    attacks = ["fgsm", "pgd"]

    fig, axes = plt.subplots(len(xai_methods), len(attacks), figsize=(16, 6 * len(xai_methods)))
    if len(xai_methods) == 1:
        axes = axes.reshape(1, -1)

    for row, xai in enumerate(xai_methods):
        for col, atk in enumerate(attacks):
            ax = axes[row, col]
            if (xai, atk) not in attribution_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            data = attribution_data[(xai, atk)]
            attr_clean = np.asarray(data["attr_clean"])
            attr_adv = np.asarray(data["attr_adv"])

            # SHAP may return (N, D, classes) — flatten to (N, D)
            if attr_clean.ndim > 2:
                attr_clean = attr_clean.reshape(attr_clean.shape[0], -1)
            if attr_adv.ndim > 2:
                attr_adv = attr_adv.reshape(attr_adv.shape[0], -1)

            mean_shift = np.abs(attr_adv - attr_clean).mean(axis=0)
            top_idx = np.argsort(mean_shift)[-k:][::-1]
            top_shift = mean_shift[top_idx]
            labels = [f"F{i}" for i in top_idx]

            ax.barh(range(k), top_shift[::-1], color=COLOURS.get(atk, COLOURS["fgsm"]),
                    edgecolor="black", linewidth=0.5)
            ax.set_yticks(range(k))
            ax.set_yticklabels(labels[::-1])
            ax.set_xlabel("Mean |Δ Attribution|")
            ax.set_title(f"{xai.upper()} + {atk.upper()}")
            ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Top-{k} Feature Attribution Shifts Under Attack", fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "fig_topk_shift_combined")


# ══════════════════════════════════════════════════════════════════════
# Pipeline Re-run (to extract raw scores)
# ══════════════════════════════════════════════════════════════════════
def run_pipeline_for_raw_scores(cfg_path, epsilon=0.01):
    """Run the pipeline and extract raw drift score arrays."""
    import yaml
    import torch
    from src.data.loader import load_dataset
    from src.models.mlp import build_mlp, train_model
    from src.explain.ig import compute_ig
    from src.explain.shap_explainer import compute_shap
    from src.drift.metrics import compute_cosine, compute_euclidean
    from src.attacks.fgsm import fgsm_attack
    from src.attacks.pgd import pgd_attack
    from src.pipeline.run_experiment import select_eval_subset, set_seed, _clean_baseline_scores

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["run"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if cfg["run"].get("use_cuda", False) and torch.cuda.is_available() else "cpu")

    print("  Loading data...")
    X_train, X_test, y_train, y_test, scaler, _ = load_dataset(cfg, return_metadata=True)

    print("  Training model...")
    model = build_mlp(cfg, input_dim=X_train.shape[1])
    model, _ = train_model(model, X_train, y_train, cfg)
    model = model.to(device)

    max_eval = int(cfg["run"].get("max_eval", 500))
    X_eval, y_eval = select_eval_subset(model, X_test, y_test, max_eval=max_eval, device=device, seed=seed)

    metric_fns = {"cosine": compute_cosine, "euclidean": compute_euclidean}

    atk_cfg = cfg.get("attack", {})
    alpha = float(atk_cfg.get("alpha", 0.005))
    iters = int(atk_cfg.get("iters", 10))

    X_bg = torch.tensor(X_train[:100], dtype=torch.float32, device=device)

    raw_scores = {}
    attribution_data = {}

    for atk_name in ("fgsm", "pgd"):
        print(f"  Generating {atk_name.upper()} adversarial examples (ε={epsilon})...")
        if atk_name == "fgsm":
            X_adv = fgsm_attack(model, X_eval, y_eval, epsilon)
        else:
            X_adv = pgd_attack(model, X_eval, y_eval, epsilon, alpha, iters)

        # Filter to preserved predictions
        model.eval()
        with torch.no_grad():
            pred_clean = torch.argmax(model(X_eval), dim=1)
            pred_adv = torch.argmax(model(X_adv), dim=1)
        mask = pred_clean == pred_adv
        X_c = X_eval[mask]
        X_a = X_adv[mask]
        y_c = y_eval[mask]

        if mask.sum() == 0:
            print(f"    WARNING: All predictions flipped for {atk_name}, skipping.")
            continue

        for xai_name in ("ig", "shap"):
            print(f"  Computing {xai_name.upper()} attributions for {atk_name.upper()}...")
            if xai_name == "ig":
                attr_clean = compute_ig(model, X_c, target=1)
                attr_adv = compute_ig(model, X_a, target=1)
            else:
                attr_clean = compute_shap(model, X_c, X_bg)
                attr_adv = compute_shap(model, X_a, X_bg)

            scores_dict = {}
            for m_name, m_fn in metric_fns.items():
                adv_scores = m_fn(attr_clean, attr_adv)
                clean_scores = _clean_baseline_scores(m_fn, attr_clean, y_c, seed)
                scores_dict[f"adv_{m_name}"] = adv_scores
                scores_dict[f"clean_{m_name}"] = clean_scores

            raw_scores[(xai_name, atk_name)] = scores_dict
            attribution_data[(xai_name, atk_name)] = {
                "attr_clean": np.asarray(attr_clean),
                "attr_adv": np.asarray(attr_adv),
            }

    return raw_scores, attribution_data


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def find_latest_sweep_csv():
    """Auto-detect the most recent sweep CSV in results/logs/."""
    results_dir = os.path.join(PROJECT_ROOT, "results", "logs")
    candidates = []
    for d in os.listdir(results_dir):
        csv_path = os.path.join(results_dir, d, "epsilon_sweep_results.csv")
        if os.path.isfile(csv_path):
            candidates.append(csv_path)
    if not candidates:
        return None
    return sorted(candidates)[-1]


def main():
    parser = argparse.ArgumentParser(description="Generate dissertation figures")
    parser.add_argument("--sweep-csv", type=str, default=None,
                        help="Path to epsilon_sweep_results.csv (auto-detected if omitted)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "legacy", "figures"),
                        help="Output directory for figures")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run pipeline to extract raw drift scores for distribution plots")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Epsilon to use when re-running pipeline (default: 0.01)")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "experiment.yaml"),
                        help="Path to experiment config YAML")
    parser.add_argument("--topk", type=int, default=15,
                        help="Number of top features for attribution shift plot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Sweep-based plots (from CSV, no re-run needed) ───────────
    sweep_csv = args.sweep_csv or find_latest_sweep_csv()
    if sweep_csv and os.path.isfile(sweep_csv):
        print(f"Loading sweep data from: {sweep_csv}")
        df = pd.read_csv(sweep_csv)
        print(f"  {len(df)} rows, epsilons: {sorted(df['epsilon'].unique())}")

        print("\nGenerating sweep-based plots...")
        plot_epsilon_vs_auc(df, args.output_dir)
        plot_epsilon_vs_drift(df, args.output_dir)
        plot_xai_comparison_bars(df, args.output_dir)
        plot_auc_heatmap(df, args.output_dir)
        plot_flip_rate(df, args.output_dir)
    else:
        print("No sweep CSV found. Skipping sweep-based plots.")
        print("  Run an epsilon sweep first, or pass --sweep-csv <path>.")

    # ── Raw-score plots (require pipeline re-run or cache) ───────
    cache_path = os.path.join(args.output_dir, "raw_scores.npz")

    if args.rerun:
        print(f"\nRe-running pipeline (ε={args.epsilon}) to extract raw scores...")
        raw_scores, attribution_data = run_pipeline_for_raw_scores(args.config, epsilon=args.epsilon)

        # Cache raw scores for future re-plotting without re-running
        save_dict = {}
        for (xai, atk), scores in raw_scores.items():
            for key, arr in scores.items():
                save_dict[f"{xai}__{atk}__{key}"] = arr
        for (xai, atk), data in attribution_data.items():
            save_dict[f"{xai}__{atk}__attr_clean"] = data["attr_clean"]
            save_dict[f"{xai}__{atk}__attr_adv"] = data["attr_adv"]
        np.savez_compressed(cache_path, **save_dict)
        print(f"  Cached raw scores to {cache_path}")

    elif os.path.isfile(cache_path):
        print(f"\nLoading cached raw scores from {cache_path}")
        loaded = np.load(cache_path)
        raw_scores = {}
        attribution_data = {}
        for full_key in loaded.files:
            parts = full_key.split("__")
            if len(parts) != 3:
                continue
            xai, atk, field = parts
            pair = (xai, atk)
            if field.startswith("attr_"):
                attribution_data.setdefault(pair, {})[field] = loaded[full_key]
            elif field.startswith("adv_") or field.startswith("clean_"):
                # Only load cosine and euclidean
                if "cosine" in field or "euclidean" in field:
                    raw_scores.setdefault(pair, {})[field] = loaded[full_key]
    else:
        raw_scores = None
        attribution_data = None

    if raw_scores:
        print("\nGenerating raw-score plots...")
        # Combined 2×2 figures (space-saving for dissertation)
        plot_clean_vs_adversarial_combined(raw_scores, args.output_dir)
        plot_roc_combined(raw_scores, args.output_dir)
        # Individual figures (for flexibility)
        plot_clean_vs_adversarial_individual(raw_scores, args.output_dir)
        plot_roc_overlay(raw_scores, args.output_dir)

    if attribution_data:
        plot_topk_feature_shift(attribution_data, args.output_dir, k=args.topk)

    if not raw_scores and not attribution_data:
        print("\nTo generate distribution/ROC/feature-shift plots, run with --rerun:")
        print(f"  python legacy/dissertation_plots.py --rerun --epsilon 0.01")

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
