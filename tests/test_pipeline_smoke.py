import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.run_experiment import run_pipeline


class PipelineSmokeTests(unittest.TestCase):
    def test_tiny_synthetic_fgsm_ig_pipeline_writes_research_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "cicids2018"
            data_dir.mkdir(parents=True)
            rng = np.random.default_rng(4)
            benign = rng.normal(loc=-2.0, scale=0.2, size=(40, 4))
            attack = rng.normal(loc=2.0, scale=0.2, size=(40, 4))
            X = np.vstack([benign, attack])
            labels = ["Benign"] * len(benign) + ["Attack"] * len(attack)
            df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
            df["Label"] = labels
            df.to_csv(data_dir / "synthetic.csv", index=False)

            cfg = {
                "run": {"seed": 4, "use_cuda": False, "output_dir": str(root / "results")},
                "data": {
                    "dataset": "cicids2018",
                    "root": str(root / "data"),
                    "test_size": 0.25,
                    "cicids2018": {"label_col": "Label", "benign_label": "Benign"},
                },
                "model": {
                    "hidden_dims": [8],
                    "dropout": 0.0,
                    "num_classes": 2,
                    "epochs": 12,
                    "batch_size": 16,
                    "lr": 0.01,
                    "patience": 4,
                },
                "attack": {"epsilon": 0.01, "alpha": 0.005, "iters": 2},
                "explain": {
                    "ig": {"batch_size": 8, "n_steps": 4},
                    "shap": {"background_size": 8, "batch_size": 8},
                },
            }

            results = run_pipeline(
                cfg,
                csv_filename="synthetic.csv",
                attack_type="fgsm",
                xai_method="ig",
                max_eval=10,
            )

            out_dir = Path(results["out_dir"])
            self.assertTrue((out_dir / "metrics_summary.json").exists())
            self.assertTrue((out_dir / "metrics_schema.json").exists())
            self.assertTrue((out_dir / "run_metadata.json").exists())
            self.assertTrue((out_dir / "experiment_summary.csv").exists())
            self.assertIn("run_metadata", results)
            self.assertIn("metrics_schema", results)
            self.assertIn("summary_path", results)
            self.assertIn("model_metrics", results["metrics_schema"])
            self.assertIn("baseline_metrics", results["metrics_schema"])
            self.assertIn("ig_fgsm", results["metrics"])
            self.assertIn("mean_clean_cosine", results["metrics"]["ig_fgsm"])

            metadata = json.loads((out_dir / "run_metadata.json").read_text())
            self.assertEqual(metadata["data_metadata"]["feature_count"], 4)
            self.assertGreaterEqual(metadata["evaluation_subset"]["actual_total"], 2)
            self.assertEqual(metadata["resolved_config"]["attack"]["type"], "fgsm")
            self.assertEqual(metadata["resolved_config"]["explain"]["method"], "ig")

            summary = pd.read_csv(out_dir / "experiment_summary.csv")
            self.assertEqual(len(summary), 2)
            self.assertEqual(set(summary["drift_metric"]), {"cosine", "euclidean"})
            self.assertIn("baseline_confidence_auc", summary.columns)
            self.assertIn("mean_clean_drift", summary.columns)
            self.assertEqual(summary["xai"].unique().tolist(), ["ig"])
            self.assertEqual(summary["attack"].unique().tolist(), ["fgsm"])


if __name__ == "__main__":
    unittest.main()
