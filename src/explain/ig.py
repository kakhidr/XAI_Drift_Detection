import numpy as np
import torch
from captum.attr import IntegratedGradients


def compute_ig(model, X, target: int = 1, batch_size: int = 64, n_steps: int = 50):
    """
    Compute Integrated Gradients attributions.

    Returns numpy array of shape [N, D].
    """
    model.eval()
    ig = IntegratedGradients(model)
    attributions = []

    for i in range(0, X.shape[0], batch_size):
        batch = X[i : i + batch_size].clone().requires_grad_(True)
        attr = ig.attribute(batch, target=target, n_steps=n_steps)
        attributions.append(attr.detach().cpu().numpy())

    return np.concatenate(attributions, axis=0)