def compute_roc(scores, out_dir: str, name: str = "metric"):
    """
    Compute ROC-AUC for 'scores' against a binary target:
    - preserved (0) vs not preserved (1), or another defined label strategy.
    Save a plot to out_dir.
    Return AUC float.
    """
    raise NotImplementedError