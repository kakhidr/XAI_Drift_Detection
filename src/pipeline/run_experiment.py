import os
import json
import datetime
import platform
import importlib.metadata
import tempfile
import copy
import numpy as np
import torch

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "xai_drift_cache")
_MPL_DIR = os.path.join(tempfile.gettempdir(), "xai_drift_matplotlib")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.makedirs(_MPL_DIR, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", _MPL_DIR)

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


def _label_counts(values) -> dict[str, int]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    vals, counts = np.unique(np.asarray(values), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _package_versions():
    packages = ["torch", "numpy", "pandas", "scikit-learn", "captum", "shap", "streamlit"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _model_predictions(model, X, device):
    X_t = torch.tensor(X, dtype=torch.float32, device=device) if not hasattr(X, "detach") else X.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
    return X_t, preds, probs


def _model_metrics(model, X_test, y_test, device):
    _, preds, probs = _model_predictions(model, X_test, device)
    y_t = torch.tensor(y_test, dtype=torch.long, device=device)
    correct = preds == y_t
    return {
        "test_accuracy": float(correct.float().mean().item()),
        "test_correct": int(correct.sum().item()),
        "test_total": int(len(y_t)),
        "prediction_distribution": _label_counts(preds),
        "mean_class_1_confidence": float(probs[:, 1].mean().item()),
    }


def select_eval_subset(model, X_test, y_test, max_eval: int = 500, device="cpu",
                       seed: int = 42, return_metadata: bool = False):
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

    n_selected_each = min(n_each, len(benign_idx), len(attack_idx))
    if n_selected_each == 0:
        raise ValueError(
            "Cannot build a balanced evaluation subset: no correctly classified samples "
            "are available for at least one class."
        )

    rng = np.random.default_rng(seed)
    benign_perm = torch.as_tensor(
        rng.permutation(len(benign_idx))[:n_selected_each],
        dtype=torch.long,
        device=device,
    )
    attack_perm = torch.as_tensor(
        rng.permutation(len(attack_idx))[:n_selected_each],
        dtype=torch.long,
        device=device,
    )
    eval_idx = torch.cat([benign_idx[benign_perm], attack_idx[attack_perm]])
    shuffle_perm = torch.as_tensor(rng.permutation(len(eval_idx)), dtype=torch.long, device=device)
    eval_idx = eval_idx[shuffle_perm]
    X_eval = X_t[eval_idx]
    y_eval = y_t[eval_idx]

    metadata = {
        "requested_total": int(max_eval),
        "requested_per_class": int(n_each),
        "actual_total": int(len(eval_idx)),
        "selected_per_class": {"0": int(n_selected_each), "1": int(n_selected_each)},
        "available_correct_per_class": {"0": int(len(benign_idx)), "1": int(len(attack_idx))},
        "total_correct": int(len(correct)),
        "test_accuracy_before_subset": float(len(correct) / max(len(y_t), 1)),
        "warnings": [],
    }
    if n_selected_each < n_each:
        metadata["warnings"].append(
            "Fewer correctly classified samples were available than requested; "
            "the evaluation subset was reduced to preserve class balance."
        )

    if return_metadata:
        return X_eval, y_eval, metadata
    return X_eval, y_eval


def _clean_pair_indices(labels, seed: int):
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels)
    paired = np.arange(len(labels))
    rng = np.random.default_rng(seed)

    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if len(idx) <= 1:
            continue
        shuffled = rng.permutation(idx)
        if np.any(shuffled == idx):
            shuffled = np.roll(shuffled, 1)
        paired[idx] = shuffled
    return paired


def _clean_baseline_scores(metric_fn, attr_clean, labels, seed: int):
    attr_clean = np.asarray(attr_clean)
    if len(attr_clean) <= 1:
        return np.zeros(len(attr_clean), dtype=float)
    paired = _clean_pair_indices(labels, seed)
    return metric_fn(attr_clean, attr_clean[paired])


def _compute_detection(metric_fn, attr_clean, attr_adv, labels, out_dir, name, seed):
    adv_scores = metric_fn(attr_clean, attr_adv)
    clean_scores = _clean_baseline_scores(metric_fn, attr_clean, labels, seed)
    auc_val, fig, path, details = compute_roc(
        adv_scores,
        out_dir,
        name=name,
        clean_scores=clean_scores,
        seed=seed,
        return_details=True,
    )
    return {
        "auc": float(auc_val),
        "figure": fig,
        "path": path,
        "details": details,
        "clean_scores": clean_scores,
        "adversarial_scores": adv_scores,
    }


def _confidence_baseline(model, X_clean, X_adv, labels, seed):
    model.eval()
    with torch.no_grad():
        clean_probs = torch.softmax(model(X_clean), dim=1)[:, 1].detach().cpu().numpy()
        adv_probs = torch.softmax(model(X_adv), dim=1)[:, 1].detach().cpu().numpy()
    paired = _clean_pair_indices(labels, seed)
    return np.abs(clean_probs - clean_probs[paired]), np.abs(adv_probs - clean_probs)


def _input_norm_baseline(X_clean, X_adv, labels, seed):
    clean_np = X_clean.detach().cpu().numpy()
    adv_np = X_adv.detach().cpu().numpy()
    paired = _clean_pair_indices(labels, seed)
    clean_scores = np.linalg.norm(clean_np - clean_np[paired], axis=1)
    adv_scores = np.linalg.norm(adv_np - clean_np, axis=1)
    return clean_scores, adv_scores


def _baseline_roc(clean_scores, adv_scores, out_dir, name, seed):
    auc_val, _, path, details = compute_roc(
        adv_scores,
        out_dir,
        name=name,
        clean_scores=clean_scores,
        seed=seed,
        return_details=True,
    )
    return {"auc": float(auc_val), "path": path, **details}


def _run_metadata(cfg, csv_filename, out_dir, data_metadata, model_metrics, eval_metadata,
                  device, attack_type, xai_method, max_eval):
    resolved_cfg = copy.deepcopy(cfg)
    resolved_cfg.setdefault("attack", {})["type"] = attack_type
    resolved_cfg.setdefault("explain", {})["method"] = xai_method
    return {
        "run_id": os.path.basename(out_dir),
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "csv_filename": csv_filename,
        "resolved_config": _json_safe(resolved_cfg),
        "seed": int(cfg["run"]["seed"]),
        "device": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "data_metadata": data_metadata,
        "model_metrics": model_metrics,
        "evaluation_subset": eval_metadata,
        "requested_attack_type": attack_type,
        "requested_xai_method": xai_method,
        "requested_max_eval": int(max_eval),
    }


def _save_json(path, payload):
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=4)


def _parse_detection_key(key):
    parts = key.split("_")
    xai = parts[0] if len(parts) > 0 else None
    attack = parts[1] if len(parts) > 1 else None
    epsilon = None
    if "eps" in parts:
        eps_idx = parts.index("eps")
        if eps_idx + 1 < len(parts):
            try:
                epsilon = float(parts[eps_idx + 1])
            except ValueError:
                epsilon = parts[eps_idx + 1]
    return xai, attack, epsilon


def _ci_bounds(details):
    ci = details.get("auc_ci_95") if isinstance(details, dict) else None
    if not ci:
        return None, None
    return ci.get("low"), ci.get("high")


def _threshold_value(details, threshold_name, field):
    thresholds = details.get("threshold_metrics", {}) if isinstance(details, dict) else {}
    return thresholds.get(threshold_name, {}).get(field)


def _lookup_summary_metric(legacy_metrics, sweep_results, xai, attack, epsilon, metric):
    if sweep_results is not None and epsilon is not None:
        for row in sweep_results:
            if row.get("xai") == xai and row.get("attack") == attack and float(row.get("epsilon")) == float(epsilon):
                return {
                    "mean_adversarial_drift": row.get(f"mean_{metric}_drift"),
                    "mean_clean_drift": row.get(f"mean_clean_{metric}_drift"),
                    "n_preserved": row.get("preserved_count"),
                    "n_flipped": row.get("flip_count"),
                }
    vals = legacy_metrics.get(f"{xai}_{attack}", {})
    return {
        "mean_adversarial_drift": vals.get(f"mean_{metric}"),
        "mean_clean_drift": vals.get(f"mean_clean_{metric}"),
        "n_preserved": vals.get("n_preserved"),
        "n_flipped": vals.get("n_flipped"),
    }


def _baseline_key(attack, epsilon):
    if epsilon is None:
        return attack
    return f"{attack}_eps_{epsilon}"


def write_experiment_summary(out_dir, run_metadata, metrics_schema, legacy_metrics=None, sweep_results=None):
    import pandas as pd

    legacy_metrics = legacy_metrics or {}
    rows = []
    data_metadata = run_metadata.get("data_metadata", {})
    model_metrics = metrics_schema.get("model_metrics", {})
    attack_metrics = metrics_schema.get("attack_metrics", {})
    baseline_metrics = metrics_schema.get("baseline_metrics", {})
    drift_metrics = metrics_schema.get("drift_detection_metrics", {})

    for detection_key, metrics_by_name in drift_metrics.items():
        xai, attack, epsilon = _parse_detection_key(detection_key)
        atk_key = _baseline_key(attack, epsilon)
        attack_info = attack_metrics.get(atk_key, attack_metrics.get(attack, {}))
        baselines = baseline_metrics.get(atk_key, baseline_metrics.get(attack, {})) or {}
        confidence_baseline = baselines.get("prediction_confidence") or {}
        input_baseline = baselines.get("input_l2_norm") or {}
        random_null = metrics_by_name.get("random_attribution_null") or {}

        for metric_name in ("cosine", "euclidean", "kl"):
            details = metrics_by_name.get(metric_name)
            if not isinstance(details, dict):
                continue
            ci_low, ci_high = _ci_bounds(details)
            summary_metric = _lookup_summary_metric(
                legacy_metrics, sweep_results, xai, attack, epsilon, metric_name
            )
            rows.append({
                "run_id": run_metadata.get("run_id"),
                "created_at_utc": run_metadata.get("created_at_utc"),
                "dataset": data_metadata.get("dataset"),
                "csv_filename": run_metadata.get("csv_filename"),
                "row_count": data_metadata.get("row_count"),
                "feature_count": data_metadata.get("feature_count"),
                "xai": xai,
                "attack": attack,
                "epsilon": attack_info.get("epsilon", epsilon),
                "pgd_alpha": attack_info.get("alpha"),
                "pgd_iters": attack_info.get("iters"),
                "drift_metric": metric_name,
                "auc": details.get("auc"),
                "auc_ci_low": ci_low,
                "auc_ci_high": ci_high,
                "mean_adversarial_drift": summary_metric.get("mean_adversarial_drift"),
                "mean_clean_drift": summary_metric.get("mean_clean_drift"),
                "clean_score_mean": details.get("clean_score_mean"),
                "adversarial_score_mean": details.get("adversarial_score_mean"),
                "threshold_clean_p95": _threshold_value(details, "clean_p95", "threshold"),
                "fpr_at_clean_p95": _threshold_value(details, "clean_p95", "fpr"),
                "tpr_at_clean_p95": _threshold_value(details, "clean_p95", "tpr"),
                "threshold_clean_p99": _threshold_value(details, "clean_p99", "threshold"),
                "fpr_at_clean_p99": _threshold_value(details, "clean_p99", "fpr"),
                "tpr_at_clean_p99": _threshold_value(details, "clean_p99", "tpr"),
                "n_preserved": summary_metric.get("n_preserved", attack_info.get("n_preserved")),
                "n_flipped": summary_metric.get("n_flipped", attack_info.get("n_flipped")),
                "flip_rate": attack_info.get("flip_rate"),
                "model_test_accuracy": model_metrics.get("test_accuracy"),
                "model_test_correct": model_metrics.get("test_correct"),
                "model_test_total": model_metrics.get("test_total"),
                "eval_requested_total": run_metadata.get("evaluation_subset", {}).get("requested_total"),
                "eval_actual_total": run_metadata.get("evaluation_subset", {}).get("actual_total"),
                "baseline_confidence_auc": confidence_baseline.get("auc"),
                "baseline_input_l2_auc": input_baseline.get("auc"),
                "baseline_random_attr_auc": random_null.get("auc"),
                "device": run_metadata.get("device"),
                "seed": run_metadata.get("seed"),
            })

    summary_path = os.path.join(out_dir, "experiment_summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


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
    X_train, X_test, y_train, y_test, scaler, data_metadata = load_dataset(
        cfg, csv_filename, return_metadata=True
    )
    timer.stop()

    # --- 2. Model Training ---
    status("Building and training model...")
    timer.start("Model Training")
    model = build_mlp(cfg, input_dim=X_train.shape[1])
    model, history = train_model(model, X_train, y_train, cfg, progress_cb=train_cb)
    model = model.to(device)
    timer.stop()
    model_metrics = _model_metrics(model, X_test, y_test, device)

    # --- 3. Evaluation Subset ---
    status(f"Selecting {max_eval} evaluation samples...")
    timer.start("Eval Subset Selection")
    X_eval, y_eval, eval_metadata = select_eval_subset(
        model, X_test, y_test, max_eval=max_eval, device=device, seed=seed, return_metadata=True
    )
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
    attack_metrics = {}
    drift_detection_metrics = {}
    baseline_metrics = {}

    for atk, pdata in preserved.items():
        X_c = pdata["X_clean"]
        X_a = pdata["X_adv"]
        y_c = pdata["y"]
        total_attack_samples = pdata["n_preserved"] + pdata["n_flipped"]
        attack_metrics[atk] = {
            "epsilon": epsilon,
            "alpha": alpha if atk == "pgd" else None,
            "iters": iters if atk == "pgd" else None,
            "n_preserved": pdata["n_preserved"],
            "n_flipped": pdata["n_flipped"],
            "flip_rate": float(pdata["n_flipped"] / max(total_attack_samples, 1)),
        }

        if pdata["n_preserved"] == 0:
            baseline_metrics[atk] = {
                "prediction_confidence": None,
                "input_l2_norm": None,
            }
            for method in xai_methods:
                key = f"{method}_{atk}"
                all_metrics[key] = {
                    "auc_cosine": float("nan"),
                    "auc_euclidean": float("nan"),
                    "auc_kl": float("nan"),
                    "mean_cosine": float("nan"),
                    "mean_clean_cosine": float("nan"),
                    "mean_euclidean": float("nan"),
                    "mean_clean_euclidean": float("nan"),
                    "mean_kl": float("nan"),
                    "mean_clean_kl": float("nan"),
                    "n_preserved": 0,
                    "n_flipped": pdata["n_flipped"],
                }
            continue

        conf_clean, conf_adv = _confidence_baseline(model, X_c, X_a, y_c, seed)
        input_clean, input_adv = _input_norm_baseline(X_c, X_a, y_c, seed)
        baseline_metrics[atk] = {
            "prediction_confidence": _baseline_roc(
                conf_clean, conf_adv, out_dir, f"baseline_confidence_{atk}", seed
            ),
            "input_l2_norm": _baseline_roc(
                input_clean, input_adv, out_dir, f"baseline_input_l2_{atk}", seed
            ),
        }

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
            cos_result = _compute_detection(
                compute_cosine, attr_clean, attr_adv, y_c, out_dir, f"{method}_{atk}_cos", seed
            )
            euc_result = _compute_detection(
                compute_euclidean, attr_clean, attr_adv, y_c, out_dir, f"{method}_{atk}_euc", seed
            )
            kl_result = _compute_detection(
                compute_kl, attr_clean, attr_adv, y_c, out_dir, f"{method}_{atk}_kl", seed
            )
            cos_d = cos_result["adversarial_scores"]
            euc_d = euc_result["adversarial_scores"]
            kl_d = kl_result["adversarial_scores"]
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
            rng = np.random.default_rng(seed)
            rand_clean = rng.normal(size=np.asarray(attr_clean).shape)
            rand_adv = rng.normal(size=np.asarray(attr_adv).shape)
            random_null = _compute_detection(
                compute_euclidean,
                rand_clean,
                rand_adv,
                y_c,
                out_dir,
                f"baseline_random_attr_{method}_{atk}",
                seed,
            )
            drift_detection_metrics[key] = {
                "cosine": cos_result["details"],
                "euclidean": euc_result["details"],
                "kl": kl_result["details"],
                "random_attribution_null": random_null["details"],
            }
            all_metrics[key] = {
                "auc_cosine": float(cos_result["auc"]),
                "auc_euclidean": float(euc_result["auc"]),
                "auc_kl": float(kl_result["auc"]),
                "auc_cosine_ci_low": (
                    None if cos_result["details"]["auc_ci_95"] is None
                    else float(cos_result["details"]["auc_ci_95"]["low"])
                ),
                "auc_cosine_ci_high": (
                    None if cos_result["details"]["auc_ci_95"] is None
                    else float(cos_result["details"]["auc_ci_95"]["high"])
                ),
                "mean_cosine": float(np.mean(cos_d)),
                "mean_clean_cosine": float(np.mean(cos_result["clean_scores"])),
                "mean_euclidean": float(np.mean(euc_d)),
                "mean_clean_euclidean": float(np.mean(euc_result["clean_scores"])),
                "mean_kl": float(np.mean(kl_d)),
                "mean_clean_kl": float(np.mean(kl_result["clean_scores"])),
                "n_preserved": pdata["n_preserved"],
                "n_flipped": pdata["n_flipped"],
            }
            all_figures[f"{key}_roc_cos"] = cos_result["figure"]
            all_figures[f"{key}_roc_euc"] = euc_result["figure"]
            all_figures[f"{key}_roc_kl"] = kl_result["figure"]

    # Save metrics
    run_metadata = _run_metadata(
        cfg, csv_filename, out_dir, data_metadata, model_metrics, eval_metadata,
        device, attack_type, xai_method, max_eval
    )
    metrics_schema = {
        "model_metrics": model_metrics,
        "attack_metrics": attack_metrics,
        "drift_detection_metrics": drift_detection_metrics,
        "baseline_metrics": baseline_metrics,
        "data_metadata": data_metadata,
    }
    _save_json(os.path.join(out_dir, "metrics_summary.json"), all_metrics)
    _save_json(os.path.join(out_dir, "metrics_schema.json"), metrics_schema)
    _save_json(os.path.join(out_dir, "run_metadata.json"), run_metadata)
    summary_path = write_experiment_summary(
        out_dir,
        run_metadata,
        metrics_schema,
        legacy_metrics=all_metrics,
    )

    status("Pipeline complete!")

    return {
        "metrics": all_metrics,
        "metrics_schema": metrics_schema,
        "figures": all_figures,
        "timing": timer.summary(),
        "history": history,
        "out_dir": out_dir,
        "run_metadata": run_metadata,
        "summary_path": summary_path,
    }


def run_epsilon_sweep(cfg, csv_filename=None, attack_type="fgsm", xai_method="ig",
                     eps_list=None, max_eval=500, status_cb=None, train_cb=None):
    """
    Run an epsilon sweep: train once, then loop over multiple ε values.

    Returns dict with: sweep_results (list of dicts), figures, timing, history, out_dir
    """
    timer = StageTimer()

    def status(msg):
        if status_cb:
            status_cb(msg)

    if eps_list is None:
        eps_list = [0.001, 0.005, 0.01, 0.02]

    seed = int(cfg["run"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if cfg["run"].get("use_cuda", False) and torch.cuda.is_available() else "cpu")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_sweep"
    out_dir = os.path.join(cfg["run"].get("output_dir", "results"), run_id)
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Data Loading ---
    status("Loading dataset...")
    timer.start("Data Loading")
    X_train, X_test, y_train, y_test, scaler, data_metadata = load_dataset(
        cfg, csv_filename, return_metadata=True
    )
    timer.stop()

    # --- 2. Model Training (once) ---
    status("Building and training model...")
    timer.start("Model Training")
    model = build_mlp(cfg, input_dim=X_train.shape[1])
    model, history = train_model(model, X_train, y_train, cfg, progress_cb=train_cb)
    model = model.to(device)
    timer.stop()
    model_metrics = _model_metrics(model, X_test, y_test, device)

    # --- 3. Evaluation Subset (once) ---
    status(f"Selecting {max_eval} evaluation samples...")
    timer.start("Eval Subset Selection")
    X_eval, y_eval, eval_metadata = select_eval_subset(
        model, X_test, y_test, max_eval=max_eval, device=device, seed=seed, return_metadata=True
    )
    timer.stop()

    # Determine attacks and XAI methods
    attacks_to_run = []
    if attack_type in ("fgsm", "both"):
        attacks_to_run.append("fgsm")
    if attack_type in ("pgd", "both"):
        attacks_to_run.append("pgd")

    xai_methods = []
    if xai_method in ("ig", "both"):
        xai_methods.append("ig")
    if xai_method in ("shap", "both"):
        xai_methods.append("shap")

    attack_cfg = cfg.get("attack", {})
    base_alpha = float(attack_cfg.get("alpha", 0.005))
    iters = int(attack_cfg.get("iters", 10))

    ig_cfg = cfg.get("explain", {}).get("ig", {})
    shap_cfg = cfg.get("explain", {}).get("shap", {})
    X_bg = torch.tensor(X_train[:int(shap_cfg.get("background_size", 100))], dtype=torch.float32, device=device)

    # --- 4. Pre-compute clean attributions (once per XAI method) ---
    clean_attrs = {}
    for method in xai_methods:
        status(f"Computing clean {method.upper()} attributions...")
        timer.start(f"Clean XAI ({method.upper()})")
        if method == "ig":
            clean_attrs["ig"] = compute_ig(model, X_eval, target=1,
                                           batch_size=ig_cfg.get("batch_size", 64),
                                           n_steps=ig_cfg.get("n_steps", 50))
        else:
            clean_attrs["shap"] = compute_shap(model, X_eval, X_bg,
                                               batch_size=shap_cfg.get("batch_size", 64))
        timer.stop()

    # --- 5. Sweep over epsilons ---
    sweep_results = []
    attack_metrics = {}
    drift_detection_metrics = {}
    baseline_metrics = {}

    for eps_idx, eps in enumerate(eps_list):
        status(f"Epsilon sweep: ε={eps} ({eps_idx+1}/{len(eps_list)})")

        for atk in attacks_to_run:
            timer.start(f"Attack {atk.upper()} ε={eps}")
            if atk == "fgsm":
                X_adv = fgsm_attack(model, X_eval, y_eval, eps)
            else:
                pgd_alpha = eps / 4.0 if eps > base_alpha else base_alpha
                X_adv = pgd_attack(model, X_eval, y_eval, eps, pgd_alpha, iters)
            timer.stop()

            # Preserved predictions
            model.eval()
            with torch.no_grad():
                pred_clean = torch.argmax(model(X_eval), dim=1)
                pred_adv = torch.argmax(model(X_adv), dim=1)
            mask = pred_clean == pred_adv
            n_preserved = mask.sum().item()
            n_flipped = (~mask).sum().item()
            attack_key = f"{atk}_eps_{eps}"
            attack_metrics[attack_key] = {
                "epsilon": float(eps),
                "alpha": (eps / 4.0 if atk == "pgd" and eps > base_alpha else base_alpha) if atk == "pgd" else None,
                "iters": iters if atk == "pgd" else None,
                "n_preserved": int(n_preserved),
                "n_flipped": int(n_flipped),
                "flip_rate": float(n_flipped / max(n_preserved + n_flipped, 1)),
            }

            if n_preserved == 0:
                baseline_metrics[attack_key] = {
                    "prediction_confidence": None,
                    "input_l2_norm": None,
                }
                for method in xai_methods:
                    sweep_results.append({
                        "epsilon": eps, "attack": atk, "xai": method,
                        "preserved_count": 0, "preserved_ratio": 0.0,
                        "flip_count": n_flipped,
                        "mean_cosine_drift": float("nan"),
                        "mean_euclidean_drift": float("nan"),
                        "mean_kl_drift": float("nan"),
                        "auc_cosine": float("nan"),
                        "auc_euclidean": float("nan"),
                        "auc_kl": float("nan"),
                    })
                continue

            X_c = X_eval[mask]
            X_a = X_adv[mask]
            y_c = y_eval[mask]
            conf_clean, conf_adv = _confidence_baseline(model, X_c, X_a, y_c, seed)
            input_clean, input_adv = _input_norm_baseline(X_c, X_a, y_c, seed)
            baseline_metrics[attack_key] = {
                "prediction_confidence": _baseline_roc(
                    conf_clean, conf_adv, out_dir, f"sweep_baseline_confidence_{atk}_eps{eps}", seed
                ),
                "input_l2_norm": _baseline_roc(
                    input_clean, input_adv, out_dir, f"sweep_baseline_input_l2_{atk}_eps{eps}", seed
                ),
            }

            for method in xai_methods:
                label = f"{method.upper()}+{atk.upper()} ε={eps}"
                status(f"XAI: {label}...")
                timer.start(f"XAI {label}")
                if method == "ig":
                    attr_adv = compute_ig(model, X_a, target=1,
                                          batch_size=ig_cfg.get("batch_size", 64),
                                          n_steps=ig_cfg.get("n_steps", 50))
                    attr_clean_f = clean_attrs["ig"][mask]
                else:
                    attr_adv = compute_shap(model, X_a, X_bg,
                                            batch_size=shap_cfg.get("batch_size", 64))
                    attr_clean_f = clean_attrs["shap"][mask]
                timer.stop()

                # Drift
                cos_result = _compute_detection(
                    compute_cosine, attr_clean_f, attr_adv, y_c, out_dir,
                    f"sweep_{method}_{atk}_cos_eps{eps}", seed
                )
                euc_result = _compute_detection(
                    compute_euclidean, attr_clean_f, attr_adv, y_c, out_dir,
                    f"sweep_{method}_{atk}_euc_eps{eps}", seed
                )
                kl_result = _compute_detection(
                    compute_kl, attr_clean_f, attr_adv, y_c, out_dir,
                    f"sweep_{method}_{atk}_kl_eps{eps}", seed
                )
                cos_d = cos_result["adversarial_scores"]
                euc_d = euc_result["adversarial_scores"]
                kl_d = kl_result["adversarial_scores"]
                rng = np.random.default_rng(seed)
                rand_clean = rng.normal(size=np.asarray(attr_clean_f).shape)
                rand_adv = rng.normal(size=np.asarray(attr_adv).shape)
                random_null = _compute_detection(
                    compute_euclidean,
                    rand_clean,
                    rand_adv,
                    y_c,
                    out_dir,
                    f"sweep_baseline_random_attr_{method}_{atk}_eps{eps}",
                    seed,
                )
                drift_key = f"{method}_{atk}_eps_{eps}"
                drift_detection_metrics[drift_key] = {
                    "cosine": cos_result["details"],
                    "euclidean": euc_result["details"],
                    "kl": kl_result["details"],
                    "random_attribution_null": random_null["details"],
                }

                sweep_results.append({
                    "epsilon": eps,
                    "attack": atk,
                    "xai": method,
                    "preserved_count": n_preserved,
                    "preserved_ratio": n_preserved / len(y_eval),
                    "flip_count": n_flipped,
                    "mean_cosine_drift": float(np.mean(cos_d)),
                    "mean_clean_cosine_drift": float(np.mean(cos_result["clean_scores"])),
                    "mean_euclidean_drift": float(np.mean(euc_d)),
                    "mean_clean_euclidean_drift": float(np.mean(euc_result["clean_scores"])),
                    "mean_kl_drift": float(np.mean(kl_d)),
                    "mean_clean_kl_drift": float(np.mean(kl_result["clean_scores"])),
                    "auc_cosine": float(cos_result["auc"]),
                    "auc_euclidean": float(euc_result["auc"]),
                    "auc_kl": float(kl_result["auc"]),
                })

    # Save sweep results
    import pandas as _pd
    df_sweep = _pd.DataFrame(sweep_results)
    df_sweep.to_csv(os.path.join(out_dir, "epsilon_sweep_results.csv"), index=False)

    run_metadata = _run_metadata(
        cfg, csv_filename, out_dir, data_metadata, model_metrics, eval_metadata,
        device, attack_type, xai_method, max_eval
    )
    run_metadata["epsilon_values"] = list(eps_list)
    metrics_schema = {
        "model_metrics": model_metrics,
        "attack_metrics": attack_metrics,
        "drift_detection_metrics": drift_detection_metrics,
        "baseline_metrics": baseline_metrics,
        "data_metadata": data_metadata,
    }
    _save_json(os.path.join(out_dir, "metrics_sweep.json"), sweep_results)
    _save_json(os.path.join(out_dir, "metrics_schema.json"), metrics_schema)
    _save_json(os.path.join(out_dir, "run_metadata.json"), run_metadata)
    summary_path = write_experiment_summary(
        out_dir,
        run_metadata,
        metrics_schema,
        sweep_results=sweep_results,
    )

    status("Epsilon sweep complete!")

    return {
        "sweep_results": sweep_results,
        "metrics_schema": metrics_schema,
        "timing": timer.summary(),
        "history": history,
        "out_dir": out_dir,
        "run_metadata": run_metadata,
        "summary_path": summary_path,
    }
