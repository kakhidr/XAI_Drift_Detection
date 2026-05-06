import unittest

import numpy as np
import torch

from src.pipeline.run_experiment import select_eval_subset


class SignModel(torch.nn.Module):
    def forward(self, x):
        return torch.stack([-x[:, 0], x[:, 0]], dim=1)


class EvalSubsetTests(unittest.TestCase):
    def test_balanced_seeded_subset_is_reproducible(self):
        X = np.concatenate(
            [
                np.column_stack([np.full(10, -1.0), np.arange(10)]),
                np.column_stack([np.full(10, 1.0), np.arange(10)]),
            ]
        )
        y = np.array([0] * 10 + [1] * 10)

        first_X, first_y, first_meta = select_eval_subset(
            SignModel(), X, y, max_eval=6, seed=11, return_metadata=True
        )
        second_X, second_y, second_meta = select_eval_subset(
            SignModel(), X, y, max_eval=6, seed=11, return_metadata=True
        )

        self.assertTrue(torch.equal(first_X, second_X))
        self.assertTrue(torch.equal(first_y, second_y))
        self.assertEqual(first_meta, second_meta)
        self.assertEqual(first_meta["actual_total"], 6)
        self.assertEqual(torch.bincount(first_y.cpu(), minlength=2).tolist(), [3, 3])

    def test_insufficient_correct_samples_reduces_balanced_subset(self):
        X = np.concatenate(
            [
                np.column_stack([np.full(2, -1.0), np.arange(2)]),
                np.column_stack([np.full(10, 1.0), np.arange(10)]),
            ]
        )
        y = np.array([0] * 2 + [1] * 10)

        _, y_eval, metadata = select_eval_subset(
            SignModel(), X, y, max_eval=10, seed=3, return_metadata=True
        )

        self.assertEqual(metadata["actual_total"], 4)
        self.assertEqual(torch.bincount(y_eval.cpu(), minlength=2).tolist(), [2, 2])
        self.assertTrue(metadata["warnings"])


if __name__ == "__main__":
    unittest.main()
