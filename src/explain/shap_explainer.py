import numpy as np
import os
import tempfile
import torch

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "xai_drift_cache")
_MPL_DIR = os.path.join(tempfile.gettempdir(), "xai_drift_matplotlib")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.makedirs(_MPL_DIR, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", _MPL_DIR)

import shap


def compute_shap(model, X, X_background, batch_size: int = 64):
    """
    Compute SHAP (DeepExplainer) attributions for class 1.

    Parameters
    ----------
    X : torch.Tensor
        Samples to explain.
    X_background : torch.Tensor
        Background samples for DeepExplainer.

    Returns numpy array of shape [N, D].
    """
    explainer = shap.DeepExplainer(model, X_background)
    attributions = []

    for i in range(0, X.shape[0], batch_size):
        batch = X[i : i + batch_size]
        s = explainer.shap_values(batch)
        if isinstance(s, list):
            s = s[1]  # class 1
        attributions.append(np.array(s).squeeze())

    result = np.concatenate(attributions, axis=0)
    if result.ndim == 1:
        result = result.reshape(1, -1)
    return result
