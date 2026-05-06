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


def plot_drift_histogram(distances, title: str, out_dir: str, filename: str,
                         epsilon: float | None = None, info: str = "", bins: int = 30):
    """Save a drift histogram and return the figure + path."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(distances, bins=bins, edgecolor="black", color="skyblue")
    ax.set_title(title)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")

    if epsilon is not None or info:
        text = ""
        if epsilon is not None:
            text += f"ε = {epsilon}\n"
        text += info
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig, path
