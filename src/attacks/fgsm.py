import torch
import torch.nn.functional as F


def fgsm_attack(model, X, y, epsilon: float):
    """
    Fast Gradient Sign Method attack.

    Returns adversarial tensor X_adv (same shape as X).
    """
    X_adv = X.clone().detach().requires_grad_(True)
    model.eval()
    loss = F.cross_entropy(model(X_adv), y)
    model.zero_grad()
    loss.backward()
    X_adv = X_adv + epsilon * X_adv.grad.sign()
    return X_adv.detach()