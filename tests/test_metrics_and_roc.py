import tempfile
import unittest

import numpy as np

from src.drift.metrics import compute_euclidean, compute_kl
from src.eval.roc import compute_roc


class MetricsAndRocTests(unittest.TestCase):
    def test_metrics_handle_single_vectors_and_shape_errors(self):
        self.assertEqual(compute_euclidean(np.array([1.0, 2.0]), np.array([1.0, 2.0])).shape, (1,))
        self.assertAlmostEqual(compute_euclidean(np.array([1.0, 2.0]), np.array([1.0, 2.0]))[0], 0.0)
        self.assertTrue(np.isfinite(compute_kl(np.zeros((2, 3)), np.ones((2, 3)))).all())
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            compute_euclidean(np.zeros((2, 3)), np.zeros((3, 3)))

    def test_roc_uses_clean_scores_and_reports_thresholds(self):
        clean_scores = np.array([0.01, 0.02, 0.03, 0.04])
        adv_scores = np.array([0.8, 0.9, 1.0, 1.1])

        with tempfile.TemporaryDirectory() as tmp:
            auc_val, _, path, details = compute_roc(
                adv_scores,
                tmp,
                name="toy",
                clean_scores=clean_scores,
                seed=1,
                n_bootstrap=50,
                return_details=True,
            )

        self.assertGreater(auc_val, 0.99)
        self.assertTrue(path.endswith("roc_toy.png"))
        self.assertEqual(details["baseline_type"], "clean_pair")
        self.assertIn("auc_ci_95", details)
        self.assertIn("clean_p95", details["threshold_metrics"])
        self.assertEqual(details["threshold_metrics"]["clean_p95"]["fpr"], 0.25)
        self.assertEqual(details["threshold_metrics"]["clean_p95"]["tpr"], 1.0)


if __name__ == "__main__":
    unittest.main()
