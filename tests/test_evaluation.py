"""Tests for Phase 3 evaluation pipeline changes.

Tests cover:
- aggregate_metrics_per_sequence (shared/metrics.py)
- save_predictions_csv with sequence_ids (evaluate_model.py)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from scripts.shared.metrics import aggregate_metrics_per_sequence


class TestAggregateMetricsPerSequence:
    """Tests for aggregate_metrics_per_sequence."""

    def test_basic_output_structure(self):
        """Returns (rows, summary) with correct keys."""
        preds = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        seq_ids = np.array([0, 0, 1, 1])

        rows, summary = aggregate_metrics_per_sequence(preds, targets, seq_ids)

        assert len(rows) == 2
        assert set(rows[0].keys()) == {"sequence_id", "n_samples", "mae", "rmse", "accuracy"}
        assert "n_sequences" in summary
        assert "mae_mean" in summary
        assert "rmse_mean" in summary
        assert "accuracy_mean" in summary

    def test_perfect_predictions(self):
        """All metrics should be ideal for perfect predictions."""
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        seq_ids = np.array([0, 0, 0, 1, 1, 1])

        rows, summary = aggregate_metrics_per_sequence(preds, targets, seq_ids)

        assert len(rows) == 2
        for row in rows:
            assert row["mae"] == 0.0
            assert row["rmse"] == 0.0
            assert row["accuracy"] == 100.0

        assert summary["mae_mean"] == 0.0
        assert summary["accuracy_mean"] == 100.0

    def test_known_errors(self):
        """Check metrics for known error values."""
        # Sequence 0: errors = [0.1, 0.2] -> MAE=0.15, RMSE=sqrt(0.025)=0.158
        # Sequence 1: errors = [0.0, 0.0] -> MAE=0.0, RMSE=0.0
        preds = np.array([1.1, 2.2, 3.0, 4.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        seq_ids = np.array([0, 0, 1, 1])

        rows, summary = aggregate_metrics_per_sequence(preds, targets, seq_ids, threshold=0.05)

        # Sequence 0
        seq0 = [r for r in rows if r["sequence_id"] == 0][0]
        assert seq0["n_samples"] == 2
        np.testing.assert_almost_equal(seq0["mae"], 0.15, decimal=5)
        np.testing.assert_almost_equal(seq0["rmse"], np.sqrt(0.025), decimal=5)
        assert seq0["accuracy"] == 0.0  # Both errors > 0.05

        # Sequence 1
        seq1 = [r for r in rows if r["sequence_id"] == 1][0]
        assert seq1["mae"] == 0.0
        assert seq1["accuracy"] == 100.0

    def test_accuracy_threshold(self):
        """Accuracy threshold is applied correctly."""
        # All errors are 0.04 (below default 0.05 threshold)
        preds = np.array([1.04, 2.04, 3.04, 4.04])
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        seq_ids = np.array([0, 0, 1, 1])

        rows, _ = aggregate_metrics_per_sequence(preds, targets, seq_ids, threshold=0.05)
        for row in rows:
            assert row["accuracy"] == 100.0

        # With stricter threshold
        rows, _ = aggregate_metrics_per_sequence(preds, targets, seq_ids, threshold=0.03)
        for row in rows:
            assert row["accuracy"] == 0.0

    def test_single_sequence(self):
        """Works with only one sequence."""
        preds = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.1, 2.1, 3.1])
        seq_ids = np.array([5, 5, 5])

        rows, summary = aggregate_metrics_per_sequence(preds, targets, seq_ids)

        assert len(rows) == 1
        assert rows[0]["sequence_id"] == 5
        assert rows[0]["n_samples"] == 3
        assert summary["n_sequences"] == 1
        assert summary["mae_std"] == 0.0  # Only one sequence

    def test_many_sequences(self):
        """Handles many sequences correctly."""
        n_samples = 1000
        n_sequences = 50
        rng = np.random.RandomState(42)

        targets = rng.randn(n_samples)
        preds = targets + rng.randn(n_samples) * 0.1
        seq_ids = np.repeat(np.arange(n_sequences), n_samples // n_sequences)

        rows, summary = aggregate_metrics_per_sequence(preds, targets, seq_ids)

        assert len(rows) == n_sequences
        assert summary["n_sequences"] == n_sequences
        # Each sequence should have 20 samples
        for row in rows:
            assert row["n_samples"] == 20

    def test_unequal_sequence_lengths(self):
        """Handles sequences with different number of samples."""
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        seq_ids = np.array([0, 0, 0, 1, 1])  # Seq 0: 3 samples, Seq 1: 2 samples

        rows, summary = aggregate_metrics_per_sequence(preds, targets, seq_ids)

        seq0 = [r for r in rows if r["sequence_id"] == 0][0]
        seq1 = [r for r in rows if r["sequence_id"] == 1][0]
        assert seq0["n_samples"] == 3
        assert seq1["n_samples"] == 2

    def test_2d_input(self):
        """Works with [N, 1] shaped inputs."""
        preds = np.array([[1.0], [2.0], [3.0], [4.0]])
        targets = np.array([[1.0], [2.0], [3.0], [4.0]])
        seq_ids = np.array([0, 0, 1, 1])

        rows, summary = aggregate_metrics_per_sequence(preds, targets, seq_ids)

        assert len(rows) == 2
        assert summary["mae_mean"] == 0.0

    def test_summary_statistics(self):
        """Summary mean/std/median are computed correctly."""
        # Seq 0: errors = [0.1, 0.1] -> MAE=0.1
        # Seq 1: errors = [0.3, 0.3] -> MAE=0.3
        preds = np.array([1.1, 2.1, 3.3, 4.3])
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        seq_ids = np.array([0, 0, 1, 1])

        _, summary = aggregate_metrics_per_sequence(preds, targets, seq_ids)

        np.testing.assert_almost_equal(summary["mae_mean"], 0.2, decimal=5)
        np.testing.assert_almost_equal(summary["mae_std"], 0.1, decimal=5)
        np.testing.assert_almost_equal(summary["mae_median"], 0.2, decimal=5)


class TestSavePredictionsCsv:
    """Tests for save_predictions_csv with sequence_ids."""

    def test_csv_with_sequence_ids(self, tmp_path):
        """CSV includes sequence_id column when provided."""
        import pandas as pd
        from scripts.evaluate_model import save_predictions_csv

        preds = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.1, 2.1, 3.1, 4.1])
        seq_ids = np.array([0, 0, 1, 1])

        csv_path = save_predictions_csv(
            preds, targets, "test_model", tmp_path, sequence_ids=seq_ids
        )

        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["sample_idx", "sequence_id", "y_true", "y_pred", "abs_error"]
        assert len(df) == 4
        np.testing.assert_array_equal(df["sequence_id"].values, [0, 0, 1, 1])
        np.testing.assert_array_almost_equal(df["y_true"].values, targets)
        np.testing.assert_array_almost_equal(df["y_pred"].values, preds)

    def test_csv_without_sequence_ids(self, tmp_path):
        """CSV omits sequence_id column when not provided."""
        import pandas as pd
        from scripts.evaluate_model import save_predictions_csv

        preds = np.array([1.0, 2.0])
        targets = np.array([1.1, 2.1])

        csv_path = save_predictions_csv(preds, targets, "test_model", tmp_path)

        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["sample_idx", "y_true", "y_pred", "abs_error"]

    def test_csv_abs_error_correct(self, tmp_path):
        """abs_error is computed correctly."""
        import pandas as pd
        from scripts.evaluate_model import save_predictions_csv

        preds = np.array([1.0, 2.5])
        targets = np.array([1.5, 2.0])

        csv_path = save_predictions_csv(preds, targets, "test_model", tmp_path)

        df = pd.read_csv(csv_path)
        np.testing.assert_array_almost_equal(df["abs_error"].values, [0.5, 0.5])

    def test_csv_filename(self, tmp_path):
        """CSV file is named {model_name}_predictions.csv."""
        from scripts.evaluate_model import save_predictions_csv

        preds = np.array([1.0])
        targets = np.array([1.0])

        csv_path = save_predictions_csv(preds, targets, "my_model", tmp_path)

        assert Path(csv_path).name == "my_model_predictions.csv"
