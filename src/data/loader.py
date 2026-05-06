from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def list_csv_files(data_root: str, dataset: str) -> list[str]:
    """Return sorted list of CSV filenames available for *dataset* under *data_root*."""
    folder = Path(data_root) / dataset
    if not folder.exists():
        return []
    return sorted(f.name for f in folder.glob("*.csv"))


def _counts(values) -> dict[str, int]:
    vals, counts = np.unique(values, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


def load_dataset(cfg, csv_filename: str | None = None, return_metadata: bool = False):
    """
    Load and preprocess a dataset.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler
        numpy arrays ready for model training.
    """
    dataset = cfg["data"]["dataset"]
    data_root = cfg["data"]["root"]
    test_size = cfg["data"].get("test_size", 0.3)
    seed = cfg["run"]["seed"]

    ds_cfg = cfg["data"].get(dataset, {})
    files = [csv_filename] if csv_filename else ds_cfg.get("files", [])
    if not files:
        raise ValueError(f"No CSV files configured for dataset '{dataset}'.")

    frames = []
    for fname in files:
        path = Path(data_root) / dataset / fname
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"Dataset file is empty: {path}")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError(f"No rows loaded for dataset '{dataset}'.")

    # --- Label detection & binarisation ---
    if dataset == "cicids2018":
        label_col = ds_cfg.get("label_col", "Label")
        benign = ds_cfg.get("benign_label", "Benign")
        if label_col not in df.columns:
            raise ValueError(f"CICIDS2018 dataset missing expected label column '{label_col}'.")
        y = (df[label_col].astype(str).str.strip().str.lower() != benign.lower()).astype(int)
        drop_cols = [label_col]
    else:  # beth
        if "sus" in df.columns and "evil" in df.columns:
            y = ((df["sus"] == 1) | (df["evil"] == 1)).astype(int)
            drop_cols = ["sus", "evil"]
        elif "evil" in df.columns:
            y = df["evil"].astype(int)
            drop_cols = ["evil"]
        else:
            raise ValueError("BETH dataset missing expected label columns (sus/evil).")

    raw_features = df.drop(columns=drop_cols, errors="ignore")
    X = raw_features.select_dtypes(include="number")
    non_numeric_cols = [c for c in raw_features.columns if c not in X.columns]
    if X.empty:
        raise ValueError(f"Dataset '{dataset}' has no numeric feature columns after preprocessing.")
    if y.nunique() < 2:
        raise ValueError(f"Dataset '{dataset}' must contain both benign and attack samples.")
    class_counts = _counts(y.values)
    if min(class_counts.values()) < 2:
        raise ValueError(
            f"Dataset '{dataset}' has too few samples for stratified splitting: "
            f"class counts are {class_counts}."
        )

    # Handle NaN / Inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=seed, stratify=y.values
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    metadata = {
        "dataset": dataset,
        "data_root": str(data_root),
        "files": list(files),
        "row_count": int(len(df)),
        "feature_count": int(X.shape[1]),
        "feature_columns": list(X.columns),
        "dropped_non_numeric_columns": non_numeric_cols,
        "label_distribution": _counts(y.values),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_label_distribution": _counts(y_train),
        "test_label_distribution": _counts(y_test),
        "test_fraction": float(test_size),
    }

    result = (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    if return_metadata:
        return (*result, metadata)
    return result
