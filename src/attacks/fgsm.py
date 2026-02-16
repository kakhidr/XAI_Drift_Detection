import numpy as np

def compute_cosine(attr_clean, attr_shifted) -> np.ndarray:
    """Return per-sample cosine distance array of shape [N]. Accept torch or numpy inputs."""
    raise NotImplementedError

def compute_euclidean(attr_clean, attr_shifted) -> np.ndarray:
    """Return per-sample L2 distance array of shape [N]. Accept torch or numpy inputs."""
    raise NotImplementedError