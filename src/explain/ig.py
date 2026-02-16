def compute_ig(model, X, y, cfg):
    """
    Return attributions of shape [N, D] aligned with X features.

    Requirements:
    - Use Integrated Gradients (Captum)
    - Batched computation (cfg.explain.batch_size)
    - Baseline: zeros_like(X)
    """
    raise NotImplementedError("Implement Integrated Gradients attribution.")