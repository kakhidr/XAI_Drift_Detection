import os
import tempfile

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "xai_drift_cache")
_MPL_DIR = os.path.join(tempfile.gettempdir(), "xai_drift_matplotlib")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.makedirs(_MPL_DIR, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", _MPL_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

from src.eval.plot_style import configure_plot_style, style_axis


configure_plot_style()


def _as_1d(scores):
    arr = np.asarray(scores, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _bootstrap_auc(y_labels, scores, n_bootstrap: int, seed: int):
    if n_bootstrap <= 0:
        return None

    rng = np.random.default_rng(seed)
    aucs = []
    n = len(scores)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_labels[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_labels[idx], scores[idx])
        aucs.append(auc(fpr, tpr))
    if not aucs:
        return None
    return {
        "low": float(np.percentile(aucs, 2.5)),
        "high": float(np.percentile(aucs, 97.5)),
        "n_bootstrap": int(len(aucs)),
    }


def _threshold_metrics(clean_scores, adv_scores):
    thresholds = {
        "clean_p95": float(np.percentile(clean_scores, 95)),
        "clean_p99": float(np.percentile(clean_scores, 99)),
        "youden_j": None,
    }
    y_labels = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(adv_scores))])
    scores = np.concatenate([clean_scores, adv_scores])
    fpr, tpr, roc_thresholds = roc_curve(y_labels, scores)
    j_idx = int(np.argmax(tpr - fpr))
    thresholds["youden_j"] = float(roc_thresholds[j_idx])

    result = {}
    for name, threshold in thresholds.items():
        clean_flags = clean_scores >= threshold
        adv_flags = adv_scores >= threshold
        result[name] = {
            "threshold": threshold,
            "fpr": float(np.mean(clean_flags)),
            "tpr": float(np.mean(adv_flags)),
        }
    return result


def compute_roc(drift_scores, out_dir: str, name: str = "metric", clean_scores=None,
                seed: int = 42, n_bootstrap: int = 500, return_details: bool = False):
    """
    Build ROC curve treating drift as the anomaly score.

    Labels: clean=0, adversarial=1. If clean_scores is omitted, the legacy
    zero-baseline behavior is used for backwards compatibility.
    """
    adv_scores = _as_1d(drift_scores)
    if clean_scores is None:
        clean_scores = np.zeros(len(adv_scores), dtype=float)
        baseline_type = "zero"
    else:
        clean_scores = _as_1d(clean_scores)
        baseline_type = "clean_pair"

    if len(clean_scores) == 0 or len(adv_scores) == 0:
        raise ValueError("ROC evaluation requires non-empty clean and adversarial score arrays.")

    y_labels = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(adv_scores))])
    scores = np.concatenate([clean_scores, adv_scores])

    fpr, tpr, _ = roc_curve(y_labels, scores)
    roc_auc = auc(fpr, tpr)
    ci = _bootstrap_auc(y_labels, scores, n_bootstrap=n_bootstrap, seed=seed)
    thresholds = _threshold_metrics(clean_scores, adv_scores)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(fpr, tpr, color="blue", lw=3, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {name}")
    ax.legend(loc="lower right")
    style_axis(ax)
    fig.tight_layout()

    path = os.path.join(out_dir, f"roc_{name}.png")
    fig.savefig(path)
    plt.close(fig)

    details = {
        "auc": float(roc_auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc_ci_95": ci,
        "threshold_metrics": thresholds,
        "n_clean": int(len(clean_scores)),
        "n_adversarial": int(len(adv_scores)),
        "baseline_type": baseline_type,
        "clean_score_mean": float(np.mean(clean_scores)),
        "adversarial_score_mean": float(np.mean(adv_scores)),
    }
    if return_details:
        return roc_auc, fig, path, details
    return roc_auc, fig, path


def plot_combined_xai_roc(results, out_dir, name="roc_combined_xai_drift"):
    """Plot retained ROC coordinates, without recalculating scores or ROC values.

    results maps configuration keys (e.g. ig_fgsm) to cosine/euclidean details.
    Missing configurations are omitted, rather than inventing unavailable curves.
    """
    configurations = (
        ("ig_fgsm", "IG + FGSM", "#0072B2", "-", "o", 3.2, 8),
        ("ig_pgd", "IG + PGD", "#D55E00", "--", "s", 2.4, 6),
        ("shap_fgsm", "SHAP + FGSM", "#009E73", "-.", "^", 1.7, 4),
        ("shap_pgd", "SHAP + PGD", "#CC79A7", ":", "x", 1.1, 3),
    )
    # Local settings preserve the existing individual-plot style.
    with plt.rc_context({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "axes.labelpad": 3, "axes.titlepad": 6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.45))
        for ax, metric, title in zip(
            axes, ("cosine", "euclidean"),
            ("(a) Cosine Distance", "(b) Euclidean Distance"),
        ):
            for key, label, color, linestyle, marker, width, size in configurations:
                details = results.get(key, {}).get(metric)
                if details is None:
                    continue
                # Broad lines first, then narrower dashed lines; nested hollow
                # markers reveal coincident vertices without jittering ROC data.
                count = len(details["fpr"])
                marker_indices = np.unique(
                    np.linspace(0, count - 1, min(count, 12), dtype=int)
                ).tolist()
                ax.plot(
                    details["fpr"], details["tpr"],
                    label=f"{label} (AUC = {details['auc']:.4f})",
                    color=color, linestyle=linestyle, linewidth=width,
                    marker=marker, markersize=size, markerfacecolor="none",
                    markeredgewidth=0.8, markevery=marker_indices,
                )
            ax.plot([0, 1], [0, 1], color="0.55", linestyle="--",
                    linewidth=0.7, zorder=0, label="Random classifier")
            ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                   title=title, xlim=(-0.025, 1.025), ylim=(-0.025, 1.025))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.tick_params(width=0.6, length=3)
            ax.legend(loc="lower right", frameon=False, handlelength=3.4,
                      labelspacing=0.6)
        fig.tight_layout(pad=0.6, w_pad=1.6)
        paths = {}
        for extension in ("png", "pdf"):
            paths[extension] = os.path.join(out_dir, f"{name}.{extension}")
            fig.savefig(paths[extension], dpi=400, bbox_inches="tight")
        plt.close(fig)
    return fig, paths
