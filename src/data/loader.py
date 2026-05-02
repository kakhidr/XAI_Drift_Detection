import os
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


def load_dataset(cfg, csv_filename: str | None = None):
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

    frames = []
    for fname in files:
        path = os.path.join(data_root, dataset, fname)
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)

    # --- Label detection & binarisation ---
    if dataset == "cicids2018":
        label_col = ds_cfg.get("label_col", "Label")
        benign = ds_cfg.get("benign_label", "Benign")
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

    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include="number")

    # Handle NaN / Inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=seed, stratify=y.values
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler