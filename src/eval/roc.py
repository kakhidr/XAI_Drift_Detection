import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def compute_roc(drift_scores, out_dir: str, name: str = "metric"):
    """
    Build ROC curve treating drift as the anomaly score.

    Labels: clean=0 (zeros), adversarial=1 (actual drift scores).
    Returns AUC float and saves the ROC plot.
    """
    n = len(drift_scores)
    y_labels = np.concatenate([np.zeros(n), np.ones(n)])
    scores = np.concatenate([np.zeros(n), drift_scores])

    fpr, tpr, _ = roc_curve(y_labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {name}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = os.path.join(out_dir, f"roc_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return roc_auc, fig, path