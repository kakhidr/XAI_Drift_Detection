import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.special import softmax
from scipy.stats import entropy


def _to_numpy_2d(arr):
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return arr.reshape(arr.shape[0], -1)


def compute_cosine(attr_clean, attr_adv) -> np.ndarray:
    """Per-sample cosine distance, shape [N]."""
    A = _to_numpy_2d(attr_clean)
    B = _to_numpy_2d(attr_adv)
    return np.array([cosine_distances(A[i : i + 1], B[i : i + 1])[0, 0] for i in range(A.shape[0])])


def compute_euclidean(attr_clean, attr_adv) -> np.ndarray:
    """Per-sample Euclidean distance, shape [N]."""
    A = _to_numpy_2d(attr_clean)
    B = _to_numpy_2d(attr_adv)
    return np.linalg.norm(A - B, axis=1)


def compute_kl(attr_clean, attr_adv, eps: float = 1e-10) -> np.ndarray:
    """Per-sample KL divergence (softmax normalised attributions), shape [N]."""
    A = _to_numpy_2d(attr_clean)
    B = _to_numpy_2d(attr_adv)
    kl_vals = []
    for i in range(A.shape[0]):
        p = softmax(A[i]) + eps
        q = softmax(B[i]) + eps
        kl_vals.append(entropy(p, q))
    return np.array(kl_vals)