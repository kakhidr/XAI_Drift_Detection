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
