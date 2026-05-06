import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_dataset


def _cfg(root, dataset):
    return {
        "run": {"seed": 7},
        "data": {
            "dataset": dataset,
            "root": str(root),
            "test_size": 0.5,
            "cicids2018": {"label_col": "Label", "benign_label": "Benign"},
            "beth": {},
        },
    }


class DataLoaderTests(unittest.TestCase):
    def test_cicids_loader_validates_and_records_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "cicids2018"
            folder.mkdir()
            pd.DataFrame(
                {
                    "f1": [0.0, 1.0, np.inf, 3.0, 4.0, 5.0],
                    "f2": [1, 2, 3, 4, 5, 6],
                    "proto": ["tcp", "udp", "tcp", "udp", "tcp", "udp"],
                    "Label": ["Benign", "Attack", "Benign", "Attack", "Benign", "Attack"],
                }
            ).to_csv(folder / "sample.csv", index=False)

            X_train, X_test, y_train, y_test, _, metadata = load_dataset(
                _cfg(root, "cicids2018"), "sample.csv", return_metadata=True
            )

            self.assertEqual(X_train.shape[1], 2)
            self.assertEqual(X_test.shape[1], 2)
            self.assertTrue(np.isfinite(X_train).all())
            self.assertEqual(metadata["feature_count"], 2)
            self.assertEqual(metadata["dropped_non_numeric_columns"], ["proto"])
            self.assertEqual(metadata["label_distribution"], {"0": 3, "1": 3})
            self.assertEqual(set(y_train) | set(y_test), {0, 1})

    def test_beth_sus_evil_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "beth"
            folder.mkdir()
            pd.DataFrame(
                {
                    "processId": [1, 2, 3, 4, 5, 6],
                    "sus": [0, 1, 0, 0, 1, 0],
                    "evil": [0, 0, 1, 0, 0, 1],
                }
            ).to_csv(folder / "beth.csv", index=False)

            *_, metadata = load_dataset(_cfg(root, "beth"), "beth.csv", return_metadata=True)

            self.assertEqual(metadata["label_distribution"], {"0": 2, "1": 4})

    def test_missing_label_and_single_class_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "cicids2018"
            folder.mkdir()
            pd.DataFrame({"f1": [1, 2, 3, 4]}).to_csv(folder / "missing.csv", index=False)
            with self.assertRaisesRegex(ValueError, "missing expected label column"):
                load_dataset(_cfg(root, "cicids2018"), "missing.csv")

            pd.DataFrame({"f1": [1, 2, 3, 4], "Label": ["Benign"] * 4}).to_csv(
                folder / "one_class.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "both benign and attack"):
                load_dataset(_cfg(root, "cicids2018"), "one_class.csv")

            pd.DataFrame({"f1": [1, 2, 3], "Label": ["Benign", "Benign", "Attack"]}).to_csv(
                folder / "imbalanced.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "too few samples"):
                load_dataset(_cfg(root, "cicids2018"), "imbalanced.csv")


if __name__ == "__main__":
    unittest.main()
