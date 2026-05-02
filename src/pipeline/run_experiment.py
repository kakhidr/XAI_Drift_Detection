import os
import json
import datetime
import numpy as np
import torch

from src.data.loader import load_dataset
from src.models.mlp import build_mlp, train_model
from src.explain.ig import compute_ig
from src.explain.shap_explainer import compute_shap
from src.drift.metrics import compute_cosine, compute_euclidean, compute_kl
from src.eval.roc import compute_roc
from src.eval.plots import plot_drift_histogram
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.utils.timing import StageTimer


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_eval_subset(model, X_test, y_test, max_eval: int = 500, device="cpu"):
    """Select balanced, correctly-classified evaluation subset."""
    X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_test, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_t), dim=1)

    correct = torch.where(preds == y_t)[0]
    labels = y_t[correct]

    benign_idx = correct[labels == 0]
    attack_idx = correct[labels == 1]
    n_each = max_eval // 2

    eval_idx = torch.cat([benign_idx[:n_each], attack_idx[:n_each]])
    X_eval = X_t[eval_idx]
    y_eval = y_t[eval_idx]

    return X_eval, y_eval


def run_pipeline(cfg, csv_filename=None, attack_type="fgsm", xai_method="both",
                 max_eval=500, status_cb=None, train_cb=None):
    """
    Execute the full experiment pipeline.

    Parameters
    ----------
    cfg : dict            YAML config.
    csv_filename : str    Specific CSV file to load.
    attack_type : str     "fgsm" | "pgd" | "both"
    xai_method : str      "ig" | "shap" | "both"
    max_eval : int        Number of evaluation samples (e.g. 500, 1000, 3000).
    status_cb : callable  status_cb(message: str) for UI updates.
    train_cb : callable   train_cb(epoch, total, train_loss, val_loss) for training progress.

    Returns
    -------
    dict with keys: metrics, figures, timing, history
    """
    timer = StageTimer()

    def status(msg):
        if status_cb:
            status_cb(msg)

    seed = int(cfg["run"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if cfg["run"].get("use_cuda", False) and torch.cuda.is_available() else "cpu")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg["run"].get("output_dir", "results"), run_id)
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Data Loading ---
    status("Loading dataset...")
    timer.start("Data Loading")
    X_train, X_test, y_train, y_test, scaler = load_dataset(cfg, csv_filename)
    timer.stop()

    # --- 2. Model Training ---
    status("Building and training model...")
    timer.start("Model Training")
    model = build_mlp(cfg, input_dim=X_train.shape[1])
    model, history = train_model(model, X_train, y_train, cfg, progress_cb=train_cb)
    model = model.to(device)
    timer.stop()

    # --- 3. Evaluation Subset ---
    status(f"Selecting {max_eval} evaluation samples...")
    timer.start("Eval Subset Selection")
    X_eval, y_eval = select_eval_subset(model, X_test, y_test, max_eval=max_eval, device=device)
    timer.stop()

    # --- 4. Adversarial Attacks ---
    attacks_to_run = []
    if attack_type in ("fgsm", "both"):
        attacks_to_run.append("fgsm")
    if attack_type in ("pgd", "both"):
        attacks_to_run.append("pgd")

    attack_cfg = cfg.get("attack", {})
    epsilon = float(attack_cfg.get("epsilon", 0.001))
    alpha = float(attack_cfg.get("alpha", 0.005))
    iters = int(attack_cfg.get("iters", 10))

    adv_samples = {}
    for atk in attacks_to_run:
        status(f"Generating {atk.upper()} adversarial examples...")
        timer.start(f"Attack ({atk.upper()})")
        if atk == "fgsm":
            adv_samples["fgsm"] = fgsm_attack(model, X_eval, y_eval, epsilon)
        else:
            adv_samples["pgd"] = pgd_attack(model, X_eval, y_eval, epsilon, alpha, iters)
        timer.stop()

    # --- 5. Filter to preserved predictions ---
    preserved = {}
    for atk, X_adv in adv_samples.items():
        model.eval()
        with torch.no_grad():
            pred_clean = torch.argmax(model(X_eval), dim=1)
            pred_adv = torch.argmax(model(X_adv), dim=1)
        mask = pred_clean == pred_adv
        preserved[atk] = {
            "X_clean": X_eval[mask],
            "X_adv": X_adv[mask],
            "y": y_eval[mask],
            "n_preserved": mask.sum().item(),
            "n_flipped": (~mask).sum().item(),
        }

    # --- 6. XAI + Drift ---
    xai_methods = []
    if xai_method in ("ig", "both"):
        xai_methods.append("ig")
    if xai_method in ("shap", "both"):
        xai_methods.append("shap")

    ig_cfg = cfg.get("explain", {}).get("ig", {})
    shap_cfg = cfg.get("explain", {}).get("shap", {})
    X_bg = torch.tensor(X_train[:int(shap_cfg.get("background_size", 100))], dtype=torch.float32, device=device)

    all_metrics = {}
    all_figures = {}

    for atk, pdata in preserved.items():
        X_c = pdata["X_clean"]
        X_a = pdata["X_adv"]

        for method in xai_methods:
            label = f"{method.upper()} + {atk.upper()}"

            # Compute attributions
            status(f"Computing {label} attributions...")
            timer.start(f"XAI: {label}")
            if method == "ig":
                attr_clean = compute_ig(model, X_c, target=1,
                                        batch_size=ig_cfg.get("batch_size", 64),
                                        n_steps=ig_cfg.get("n_steps", 50))
                attr_adv = compute_ig(model, X_a, target=1,
                                      batch_size=ig_cfg.get("batch_size", 64),
                                      n_steps=ig_cfg.get("n_steps", 50))
            else:  # shap
                attr_clean = compute_shap(model, X_c, X_bg,
                                          batch_size=shap_cfg.get("batch_size", 64))
                attr_adv = compute_shap(model, X_a, X_bg,
                                        batch_size=shap_cfg.get("batch_size", 64))
            timer.stop()

            # Drift computation
            status(f"Computing drift for {label}...")
            timer.start(f"Drift: {label}")
            cos_d = compute_cosine(attr_clean, attr_adv)
            euc_d = compute_euclidean(attr_clean, attr_adv)
            kl_d = compute_kl(attr_clean, attr_adv)
            timer.stop()

            # ROC curves
            status(f"Generating ROC curves for {label}...")
            timer.start(f"ROC: {label}")
            auc_cos, fig_roc_cos, _ = compute_roc(cos_d, out_dir, name=f"{method}_{atk}_cos")
            auc_euc, fig_roc_euc, _ = compute_roc(euc_d, out_dir, name=f"{method}_{atk}_euc")
            auc_kl, fig_roc_kl, _ = compute_roc(kl_d, out_dir, name=f"{method}_{atk}_kl")
            timer.stop()

            # Histograms
            timer.start(f"Plots: {label}")
            _, _ = plot_drift_histogram(cos_d, f"{label} Cosine Drift", out_dir,
                                        f"hist_{method}_{atk}_cos.png", epsilon=epsilon,
                                        info="Cosine: direction change")
            _, _ = plot_drift_histogram(euc_d, f"{label} Euclidean Drift", out_dir,
                                        f"hist_{method}_{atk}_euc.png", epsilon=epsilon,
                                        info="Euclidean: magnitude change")
            _, _ = plot_drift_histogram(kl_d, f"{label} KL Drift", out_dir,
                                        f"hist_{method}_{atk}_kl.png", epsilon=epsilon,
                                        info="KL: distribution shift")
            timer.stop()

            key = f"{method}_{atk}"
            all_metrics[key] = {
                "auc_cosine": float(auc_cos),
                "auc_euclidean": float(auc_euc),
                "auc_kl": float(auc_kl),
                "mean_cosine": float(np.mean(cos_d)),
                "mean_euclidean": float(np.mean(euc_d)),
                "mean_kl": float(np.mean(kl_d)),
                "n_preserved": pdata["n_preserved"],
                "n_flipped": pdata["n_flipped"],
            }
            all_figures[f"{key}_roc_cos"] = fig_roc_cos
            all_figures[f"{key}_roc_euc"] = fig_roc_euc
            all_figures[f"{key}_roc_kl"] = fig_roc_kl

    # Save metrics
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)

    status("Pipeline complete!")

    return {
        "metrics": all_metrics,
        "figures": all_figures,
        "timing": timer.summary(),
        "history": history,
        "out_dir": out_dir,
    }