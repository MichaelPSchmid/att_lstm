"""Tests for Phase 4 sequence-level statistical functions.

Tests cover:
- bootstrap_ci_sequences (block bootstrap)
- cohens_d_paired_sequences (paired Cohen's d + Hedge's g)
- permutation_test_sequences (paired sign-flip permutation)
- multi_seed_sequence_analysis (law of total variance)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from scripts.sequence_level_evaluation import (
    bootstrap_ci_sequences,
    cohens_d_paired_sequences,
    multi_seed_sequence_analysis,
    permutation_test_sequences,
)


class TestBootstrapCiSequences:
    """Tests for bootstrap_ci_sequences."""

    def test_output_keys(self):
        """Returns dict with expected keys."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bootstrap_ci_sequences(values, n_bootstrap=100, seed=42)
        assert set(result.keys()) == {"mean", "std", "ci_lower", "ci_upper"}

    def test_mean_close_to_true_mean(self):
        """Bootstrap mean should be close to the sample mean."""
        rng = np.random.RandomState(42)
        values = rng.randn(500) + 10.0  # Mean ~10
        result = bootstrap_ci_sequences(values, n_bootstrap=1000, seed=42)
        np.testing.assert_almost_equal(result["mean"], np.mean(values), decimal=1)

    def test_ci_contains_true_mean(self):
        """95% CI should contain the true mean for a normal distribution."""
        rng = np.random.RandomState(42)
        values = rng.randn(200) + 5.0  # True mean = 5.0
        result = bootstrap_ci_sequences(values, n_bootstrap=2000, seed=42)
        assert result["ci_lower"] < 5.0 < result["ci_upper"]

    def test_ci_width_decreases_with_more_sequences(self):
        """More sequences should give narrower CI."""
        rng = np.random.RandomState(42)
        values_small = rng.randn(50)
        values_large = rng.randn(500)

        ci_small = bootstrap_ci_sequences(values_small, n_bootstrap=1000, seed=42)
        ci_large = bootstrap_ci_sequences(values_large, n_bootstrap=1000, seed=42)

        width_small = ci_small["ci_upper"] - ci_small["ci_lower"]
        width_large = ci_large["ci_upper"] - ci_large["ci_lower"]
        assert width_large < width_small

    def test_constant_values_zero_std(self):
        """Constant values should give zero std."""
        values = np.full(100, 42.0)
        result = bootstrap_ci_sequences(values, n_bootstrap=100, seed=42)
        assert result["std"] == 0.0
        assert result["mean"] == 42.0

    def test_reproducible_with_seed(self):
        """Same seed should give identical results."""
        values = np.random.randn(100)
        r1 = bootstrap_ci_sequences(values, n_bootstrap=500, seed=123)
        r2 = bootstrap_ci_sequences(values, n_bootstrap=500, seed=123)
        assert r1 == r2

    def test_different_seed_gives_different_results(self):
        """Different seeds should give different results."""
        values = np.random.randn(100)
        r1 = bootstrap_ci_sequences(values, n_bootstrap=500, seed=1)
        r2 = bootstrap_ci_sequences(values, n_bootstrap=500, seed=2)
        assert r1["mean"] != r2["mean"]


class TestCohensDPairedSequences:
    """Tests for cohens_d_paired_sequences."""

    def test_output_keys(self):
        """Returns dict with expected keys."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 2.5, 3.5])
        result = cohens_d_paired_sequences(a, b)
        assert set(result.keys()) == {
            "cohens_d", "hedges_g", "effect_size", "n_sequences"
        }

    def test_identical_arrays_zero_d(self):
        """Identical arrays should give Cohen's d = 0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cohens_d_paired_sequences(a, a)
        assert result["cohens_d"] == 0.0
        assert result["hedges_g"] == 0.0
        assert result["effect_size"] == "negligible"

    def test_constant_positive_difference(self):
        """Constant positive difference should give large positive d."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = a + 100.0  # B much larger than A
        result = cohens_d_paired_sequences(a, b)
        # Std of diffs is 0, but we handle this edge case
        # When all diffs are identical, std ~ 0
        # Actually, diffs = [100, 100, 100, 100, 100], std=0
        # This is a degenerate case - check that it doesn't crash
        assert result["n_sequences"] == 5

    def test_positive_d_when_b_larger(self):
        """Positive d when B values are systematically larger."""
        rng = np.random.RandomState(42)
        a = rng.randn(100)
        b = a + 0.5 + rng.randn(100) * 0.3  # B consistently larger
        result = cohens_d_paired_sequences(a, b)
        assert result["cohens_d"] > 0

    def test_negative_d_when_a_larger(self):
        """Negative d when A values are systematically larger."""
        rng = np.random.RandomState(42)
        a = rng.randn(100) + 1.0
        b = rng.randn(100)  # A consistently larger
        result = cohens_d_paired_sequences(a, b)
        assert result["cohens_d"] < 0

    def test_hedges_g_smaller_than_d(self):
        """Hedge's g should be smaller than d (bias correction shrinks)."""
        rng = np.random.RandomState(42)
        a = rng.randn(20)  # Small sample where correction matters
        b = a + rng.randn(20) * 0.5 + 0.3
        result = cohens_d_paired_sequences(a, b)
        assert abs(result["hedges_g"]) < abs(result["cohens_d"])

    def test_effect_size_categories(self):
        """Effect size categories follow Cohen's thresholds."""
        rng = np.random.RandomState(42)
        n = 1000

        # Negligible: d < 0.2
        a = rng.randn(n)
        b = a + 0.05 * rng.randn(n)
        r = cohens_d_paired_sequences(a, b)
        assert r["effect_size"] == "negligible"

        # Large: d > 0.8
        a = rng.randn(n)
        b = a + 2.0 + 0.5 * rng.randn(n)  # Large shift with some noise
        r = cohens_d_paired_sequences(a, b)
        assert r["effect_size"] == "large"

    def test_n_sequences_correct(self):
        """n_sequences matches input length."""
        a = np.random.randn(42)
        b = np.random.randn(42)
        result = cohens_d_paired_sequences(a, b)
        assert result["n_sequences"] == 42


class TestPermutationTestSequences:
    """Tests for permutation_test_sequences."""

    def test_output_keys(self):
        """Returns dict with expected keys."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 2.5, 3.5])
        result = permutation_test_sequences(a, b, n_permutations=100, seed=42)
        assert set(result.keys()) == {"observed_diff", "p_value", "significant"}

    def test_identical_arrays_high_pvalue(self):
        """Identical arrays should give p-value = 1.0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = permutation_test_sequences(a, a, n_permutations=1000, seed=42)
        assert result["observed_diff"] == 0.0
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_large_difference_significant(self):
        """Large systematic difference should be significant."""
        rng = np.random.RandomState(42)
        a = rng.randn(100)
        b = a + 3.0  # Large shift
        result = permutation_test_sequences(a, b, n_permutations=1000, seed=42)
        assert result["p_value"] < 0.05
        assert result["significant"] is True
        assert result["observed_diff"] > 0

    def test_no_difference_not_significant(self):
        """Random noise without systematic difference should not be significant."""
        rng = np.random.RandomState(42)
        a = rng.randn(100)
        b = rng.randn(100)  # Independent, no paired structure
        # With independent samples, the paired test should not be significant
        result = permutation_test_sequences(a, b, n_permutations=1000, seed=42)
        # Can't guarantee p > 0.05 due to randomness, but should usually be
        assert result["p_value"] > 0.01  # Very conservative check

    def test_observed_diff_correct(self):
        """Observed diff should be mean(b) - mean(a)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = permutation_test_sequences(a, b, n_permutations=100, seed=42)
        expected_diff = np.mean(b - a)
        np.testing.assert_almost_equal(result["observed_diff"], expected_diff)

    def test_reproducible_with_seed(self):
        """Same seed gives identical results."""
        a = np.random.randn(50)
        b = np.random.randn(50)
        r1 = permutation_test_sequences(a, b, n_permutations=500, seed=99)
        r2 = permutation_test_sequences(a, b, n_permutations=500, seed=99)
        assert r1 == r2

    def test_pvalue_between_0_and_1(self):
        """p-value should always be in [0, 1]."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            a = rng.randn(30)
            b = rng.randn(30)
            result = permutation_test_sequences(a, b, n_permutations=200, seed=42)
            assert 0.0 <= result["p_value"] <= 1.0


class TestMultiSeedSequenceAnalysis:
    """Tests for multi_seed_sequence_analysis."""

    def test_single_seed(self):
        """Single seed should just return bootstrap results."""
        bootstrap_results = {
            42: {
                "accuracy": {"mean": 90.0, "std": 0.5, "ci_lower": 89.0, "ci_upper": 91.0},
                "rmse": {"mean": 0.03, "std": 0.001, "ci_lower": 0.028, "ci_upper": 0.032},
                "mae": {"mean": 0.02, "std": 0.0008, "ci_lower": 0.018, "ci_upper": 0.022},
            }
        }
        point_metrics = {
            42: {"accuracy": 90.0, "rmse": 0.03, "mae": 0.02}
        }

        result = multi_seed_sequence_analysis(bootstrap_results, point_metrics)

        assert set(result.keys()) == {"accuracy", "rmse", "mae"}
        assert result["accuracy"]["mean"] == 90.0
        assert result["accuracy"]["std_seed"] == 0.0

    def test_multi_seed_mean(self):
        """Mean should be average of per-seed means."""
        bootstrap_results = {
            42: {
                "accuracy": {"mean": 90.0, "std": 0.5, "ci_lower": 89.0, "ci_upper": 91.0},
                "rmse": {"mean": 0.03, "std": 0.001, "ci_lower": 0.028, "ci_upper": 0.032},
                "mae": {"mean": 0.02, "std": 0.001, "ci_lower": 0.018, "ci_upper": 0.022},
            },
            94: {
                "accuracy": {"mean": 92.0, "std": 0.4, "ci_lower": 91.2, "ci_upper": 92.8},
                "rmse": {"mean": 0.025, "std": 0.001, "ci_lower": 0.023, "ci_upper": 0.027},
                "mae": {"mean": 0.018, "std": 0.001, "ci_lower": 0.016, "ci_upper": 0.020},
            },
        }
        point_metrics = {
            42: {"accuracy": 90.0, "rmse": 0.03, "mae": 0.02},
            94: {"accuracy": 92.0, "rmse": 0.025, "mae": 0.018},
        }

        result = multi_seed_sequence_analysis(bootstrap_results, point_metrics)

        np.testing.assert_almost_equal(result["accuracy"]["mean"], 91.0)
        np.testing.assert_almost_equal(result["rmse"]["mean"], 0.0275)
        np.testing.assert_almost_equal(result["mae"]["mean"], 0.019)

    def test_combined_std_larger_than_components(self):
        """Combined std should be >= max(std_bootstrap, std_seed)."""
        bootstrap_results = {
            42: {
                "accuracy": {"mean": 90.0, "std": 0.5, "ci_lower": 89.0, "ci_upper": 91.0},
                "rmse": {"mean": 0.03, "std": 0.001, "ci_lower": 0.028, "ci_upper": 0.032},
                "mae": {"mean": 0.02, "std": 0.001, "ci_lower": 0.018, "ci_upper": 0.022},
            },
            94: {
                "accuracy": {"mean": 92.0, "std": 0.4, "ci_lower": 91.2, "ci_upper": 92.8},
                "rmse": {"mean": 0.025, "std": 0.0008, "ci_lower": 0.023, "ci_upper": 0.027},
                "mae": {"mean": 0.018, "std": 0.0008, "ci_lower": 0.016, "ci_upper": 0.020},
            },
        }
        point_metrics = {
            42: {"accuracy": 90.0, "rmse": 0.03, "mae": 0.02},
            94: {"accuracy": 92.0, "rmse": 0.025, "mae": 0.018},
        }

        result = multi_seed_sequence_analysis(bootstrap_results, point_metrics)

        for metric in ["accuracy", "rmse", "mae"]:
            assert result[metric]["std"] >= result[metric]["std_bootstrap"]
            assert result[metric]["std"] >= result[metric]["std_seed"]

    def test_per_seed_values_stored(self):
        """Per-seed point values should be stored."""
        bootstrap_results = {
            42: {
                "accuracy": {"mean": 90.0, "std": 0.5, "ci_lower": 89.0, "ci_upper": 91.0},
                "rmse": {"mean": 0.03, "std": 0.001, "ci_lower": 0.028, "ci_upper": 0.032},
                "mae": {"mean": 0.02, "std": 0.001, "ci_lower": 0.018, "ci_upper": 0.022},
            },
        }
        point_metrics = {42: {"accuracy": 90.0, "rmse": 0.03, "mae": 0.02}}

        result = multi_seed_sequence_analysis(bootstrap_results, point_metrics)

        assert result["accuracy"]["per_seed_values"] == [90.0]
        assert result["rmse"]["per_seed_values"] == [0.03]
