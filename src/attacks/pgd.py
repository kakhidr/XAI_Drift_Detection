import torch
import torch.nn.functional as F


def pgd_attack(model, X, y, epsilon: float, alpha: float, iters: int):
    """
    Projected Gradient Descent attack.

    Returns adversarial tensor X_adv (same shape as X).
    """
    X_adv = X.clone().detach()
    X_orig = X.clone().detach()

    for _ in range(iters):
        X_adv.requires_grad_(True)
        loss = F.cross_entropy(model(X_adv), y)
        model.zero_grad()
        loss.backward()
        X_adv = X_adv + alpha * X_adv.grad.sign()
        # Project back to epsilon-ball
        X_adv = torch.min(torch.max(X_adv, X_orig - epsilon), X_orig + epsilon).detach()

    return X_adv