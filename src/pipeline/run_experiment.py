import yaml
import os
import json
import datetime
import numpy as np
import torch

from src.data.loader import load_dataset
from src.models.mlp import build_mlp, train_model
from src.explain.ig import compute_ig
from src.drift.metrics import compute_cosine, compute_euclidean
from src.eval.roc import compute_roc

# Optional attacks (only used if cfg.attack.type != none)
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_prediction_preserved(model, X_test, y_test, cfg):
    """Select an evaluation subset where the model predicts correctly (contract)."""
    device = next(model.parameters()).device
    max_eval = int(cfg.get("run", {}).get("max_eval", 500))

    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        preds = torch.argmax(logits, dim=1)

    correct_indices = torch.where(preds == y_test_t)[0]
    if len(correct_indices) == 0:
        raise RuntimeError("No correctly classified samples found for preserved evaluation set.")

    eval_indices = correct_indices[:max_eval]
    X_eval = X_test_t[eval_indices]
    y_eval = y_test_t[eval_indices]

    # contract check
    preds_eval = preds[eval_indices]
    assert torch.all(preds_eval == y_eval), "Eval set contains misclassified samples!"

    return X_eval, y_eval


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["run"]["seed"]))

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg["run"]["output_dir"], run_id)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(cfg, f)

    X_train, X_test, y_train, y_test = load_dataset(cfg)

    model = build_mlp(cfg, input_dim=X_train.shape[1])
    model = train_model(model, X_train, y_train, cfg)

    X_eval, y_eval = select_prediction_preserved(model, X_test, y_test, cfg)

    attack_type = cfg.get("attack", {}).get("type", "none").lower()
    if attack_type == "fgsm":
        X_adv = fgsm_attack(model, X_eval, y_eval, float(cfg["attack"]["epsilon"]))
    elif attack_type == "pgd":
        X_adv = pgd_attack(model, X_eval, y_eval, cfg)
    else:
        X_adv = X_eval.clone()

    attr_clean = compute_ig(model, X_eval, y_eval, cfg)
    attr_adv = compute_ig(model, X_adv, y_eval, cfg)

    cos_drift = compute_cosine(attr_clean, attr_adv)
    euc_drift = compute_euclidean(attr_clean, attr_adv)

    auc_cos = compute_roc(cos_drift, out_dir, name="cosine")
    auc_euc = compute_roc(euc_drift, out_dir, name="euclidean")

    metrics = {
        "auc_cosine": float(auc_cos),
        "auc_euclidean": float(auc_euc),
        "mean_cosine_drift": float(np.mean(cos_drift)),
        "mean_euclidean_drift": float(np.mean(euc_drift)),
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("Run completed:", run_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)