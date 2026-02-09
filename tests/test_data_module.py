"""Tests for TimeSeriesDataModule with sequence_ids support."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from model.data_module import TimeSeriesDataModule, load_data_file


def _create_dataset(tmp_path, n_sequences=20, samples_per_seq=50):
    """Helper to create test dataset files.

    Args:
        tmp_path: Temporary directory.
        n_sequences: Number of sequences.
        samples_per_seq: Samples per sequence (uniform).

    Returns:
        Dict with paths and metadata.
    """
    rng = np.random.RandomState(42)
    n_samples = n_sequences * samples_per_seq
    window_size = 5
    n_features = 3

    features = rng.randn(n_samples, window_size, n_features).astype(np.float64)
    targets = rng.randn(n_samples, 1).astype(np.float64)
    sequence_ids = []
    for seq_id in range(n_sequences):
        sequence_ids.extend([seq_id] * samples_per_seq)

    features_path = tmp_path / "features.npy"
    targets_path = tmp_path / "targets.npy"
    seq_ids_path = tmp_path / "sequence_ids.pkl"

    np.save(features_path, features)
    np.save(targets_path, targets)
    with open(seq_ids_path, "wb") as f:
        pickle.dump(sequence_ids, f)

    return {
        "features_path": str(features_path),
        "targets_path": str(targets_path),
        "sequence_ids_path": str(seq_ids_path),
        "n_samples": n_samples,
        "n_sequences": n_sequences,
        "samples_per_seq": samples_per_seq,
        "sequence_ids": sequence_ids,
    }


@pytest.fixture
def sample_data(tmp_path):
    """Small dataset: 3 sequences, 30 samples total."""
    return _create_dataset(tmp_path, n_sequences=3, samples_per_seq=10)


@pytest.fixture
def split_data(tmp_path):
    """Larger dataset: 20 sequences, 1000 samples total.

    Large enough for meaningful 70/20/10 split proportions.
    """
    return _create_dataset(tmp_path, n_sequences=20, samples_per_seq=50)


class TestSequenceIdsLoading:
    """Tests for Phase 1a: sequence_ids loading in DataModule."""

    def test_init_stores_sequence_ids_path(self, sample_data):
        """DataModule stores sequence_ids_path."""
        dm = TimeSeriesDataModule(
            feature_path=sample_data["features_path"],
            target_path=sample_data["targets_path"],
            sequence_ids_path=sample_data["sequence_ids_path"],
            batch_size=4,
        )
        assert dm.sequence_ids_path == sample_data["sequence_ids_path"]

    def test_init_requires_sequence_ids_path(self, sample_data):
        """DataModule requires sequence_ids_path parameter."""
        with pytest.raises(TypeError):
            TimeSeriesDataModule(
                feature_path=sample_data["features_path"],
                target_path=sample_data["targets_path"],
                batch_size=4,
            )

    def test_sequence_ids_loaded_after_setup(self, sample_data):
        """sequence_ids are loaded as numpy array after setup()."""
        dm = TimeSeriesDataModule(
            feature_path=sample_data["features_path"],
            target_path=sample_data["targets_path"],
            sequence_ids_path=sample_data["sequence_ids_path"],
            batch_size=4,
        )
        dm.setup()

        assert dm.sequence_ids is not None
        assert isinstance(dm.sequence_ids, np.ndarray)

    def test_sequence_ids_length_matches_samples(self, sample_data):
        """sequence_ids has same length as features/targets."""
        dm = TimeSeriesDataModule(
            feature_path=sample_data["features_path"],
            target_path=sample_data["targets_path"],
            sequence_ids_path=sample_data["sequence_ids_path"],
            batch_size=4,
        )
        dm.setup()

        assert len(dm.sequence_ids) == sample_data["n_samples"]
        assert len(dm.sequence_ids) == len(dm.X)

    def test_sequence_ids_values_correct(self, sample_data):
        """sequence_ids values match the original data."""
        dm = TimeSeriesDataModule(
            feature_path=sample_data["features_path"],
            target_path=sample_data["targets_path"],
            sequence_ids_path=sample_data["sequence_ids_path"],
            batch_size=4,
        )
        dm.setup()

        expected = np.array(sample_data["sequence_ids"])
        np.testing.assert_array_equal(dm.sequence_ids, expected)

    def test_sequence_ids_dtype_is_int(self, sample_data):
        """sequence_ids should be integer type."""
        dm = TimeSeriesDataModule(
            feature_path=sample_data["features_path"],
            target_path=sample_data["targets_path"],
            sequence_ids_path=sample_data["sequence_ids_path"],
            batch_size=4,
        )
        dm.setup()

        assert np.issubdtype(dm.sequence_ids.dtype, np.integer)

    def test_setup_still_creates_splits(self, sample_data):
        """setup() still creates train/val/test datasets when sequence_ids provided."""
        dm = TimeSeriesDataModule(
            feature_path=sample_data["features_path"],
            target_path=sample_data["targets_path"],
            sequence_ids_path=sample_data["sequence_ids_path"],
            batch_size=4,
        )
        dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None
        total = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
        assert total == sample_data["n_samples"]

    def test_missing_sequence_ids_file_raises(self, sample_data):
        """Raises FileNotFoundError if sequence_ids file doesn't exist."""
        dm = TimeSeriesDataModule(
            feature_path=sample_data["features_path"],
            target_path=sample_data["targets_path"],
            sequence_ids_path="/nonexistent/path/seq_ids.pkl",
            batch_size=4,
        )
        with pytest.raises(FileNotFoundError):
            dm.setup()


class TestSequenceLevelSplit:
    """Tests for Phase 1b: sequence-level train/val/test split."""

    def _make_dm(self, data, split_seed=0):
        dm = TimeSeriesDataModule(
            feature_path=data["features_path"],
            target_path=data["targets_path"],
            sequence_ids_path=data["sequence_ids_path"],
            batch_size=4,
            split_seed=split_seed,
        )
        dm.setup()
        return dm

    def test_no_sequence_overlap_between_splits(self, split_data):
        """No sequence appears in more than one split (critical: no leakage)."""
        dm = self._make_dm(split_data)

        train_seqs = set(dm.train_sequence_ids)
        val_seqs = set(dm.val_sequence_ids)
        test_seqs = set(dm.test_sequence_ids)

        assert len(train_seqs & val_seqs) == 0, "Train/val overlap"
        assert len(train_seqs & test_seqs) == 0, "Train/test overlap"
        assert len(val_seqs & test_seqs) == 0, "Val/test overlap"

    def test_all_sequences_assigned(self, split_data):
        """Every sequence is assigned to exactly one split."""
        dm = self._make_dm(split_data)

        all_assigned = set(dm.train_sequence_ids) | set(dm.val_sequence_ids) | set(dm.test_sequence_ids)
        expected = set(range(split_data["n_sequences"]))
        assert all_assigned == expected

    def test_all_samples_accounted_for(self, split_data):
        """Total samples across splits equals total dataset size."""
        dm = self._make_dm(split_data)

        total = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
        assert total == split_data["n_samples"]

    def test_split_proportions_on_sequence_level(self, split_data):
        """Split proportions are approximately 70/20/10 on sequence level."""
        dm = self._make_dm(split_data)
        n_seq = split_data["n_sequences"]

        n_train = len(dm.train_sequence_ids)
        n_val = len(dm.val_sequence_ids)
        n_test = len(dm.test_sequence_ids)

        # With 20 sequences: expect 14/4/2
        assert n_train == int(0.7 * n_seq)
        assert n_val == int(0.2 * n_seq)
        assert n_test == n_seq - n_train - n_val

    def test_samples_belong_to_correct_split(self, split_data):
        """Every sample in a split belongs to a sequence assigned to that split."""
        dm = self._make_dm(split_data)
        seq_ids = dm.sequence_ids

        train_seqs = set(dm.train_sequence_ids)
        val_seqs = set(dm.val_sequence_ids)
        test_seqs = set(dm.test_sequence_ids)

        for idx in dm.train_dataset.indices:
            assert seq_ids[idx] in train_seqs

        for idx in dm.val_dataset.indices:
            assert seq_ids[idx] in val_seqs

        for idx in dm.test_dataset.indices:
            assert seq_ids[idx] in test_seqs

    def test_deterministic_with_same_seed(self, split_data):
        """Same split_seed produces identical splits."""
        dm1 = self._make_dm(split_data, split_seed=0)
        dm2 = self._make_dm(split_data, split_seed=0)

        assert dm1.train_sequence_ids == dm2.train_sequence_ids
        assert dm1.val_sequence_ids == dm2.val_sequence_ids
        assert dm1.test_sequence_ids == dm2.test_sequence_ids

    def test_different_seed_gives_different_split(self, split_data):
        """Different split_seed produces different splits."""
        dm1 = self._make_dm(split_data, split_seed=0)
        dm2 = self._make_dm(split_data, split_seed=99)

        # With 20 sequences, different seeds should give different assignments
        assert dm1.train_sequence_ids != dm2.train_sequence_ids

    def test_split_independent_of_global_rng(self, split_data):
        """Split result does not depend on external numpy RNG state."""
        np.random.seed(12345)
        dm1 = self._make_dm(split_data, split_seed=0)

        np.random.seed(99999)
        dm2 = self._make_dm(split_data, split_seed=0)

        assert dm1.train_sequence_ids == dm2.train_sequence_ids

    def test_each_split_has_at_least_one_sequence(self, split_data):
        """Each split contains at least one sequence."""
        dm = self._make_dm(split_data)

        assert len(dm.train_sequence_ids) >= 1
        assert len(dm.val_sequence_ids) >= 1
        assert len(dm.test_sequence_ids) >= 1

    def test_default_split_seed_is_zero(self, split_data):
        """Default split_seed is 0."""
        dm = TimeSeriesDataModule(
            feature_path=split_data["features_path"],
            target_path=split_data["targets_path"],
            sequence_ids_path=split_data["sequence_ids_path"],
            batch_size=4,
        )
        assert dm.split_seed == 0


class TestGetSplitSequenceIds:
    """Tests for Phase 1c: per-sample sequence_id access via get_split_sequence_ids."""

    def _make_dm(self, data, split_seed=0):
        dm = TimeSeriesDataModule(
            feature_path=data["features_path"],
            target_path=data["targets_path"],
            sequence_ids_path=data["sequence_ids_path"],
            batch_size=4,
            split_seed=split_seed,
        )
        dm.setup()
        return dm

    def test_returns_numpy_array(self, split_data):
        """get_split_sequence_ids returns a numpy array."""
        dm = self._make_dm(split_data)
        result = dm.get_split_sequence_ids("test")
        assert isinstance(result, np.ndarray)

    def test_length_matches_split_size(self, split_data):
        """Returned array length matches number of samples in split."""
        dm = self._make_dm(split_data)

        for split_name in ("train", "val", "test"):
            dataset = getattr(dm, f"{split_name}_dataset")
            seq_ids = dm.get_split_sequence_ids(split_name)
            assert len(seq_ids) == len(dataset)

    def test_values_from_correct_sequences(self, split_data):
        """Every returned sequence_id belongs to the correct split."""
        dm = self._make_dm(split_data)

        test_ids = dm.get_split_sequence_ids("test")
        assert set(test_ids.tolist()).issubset(set(dm.test_sequence_ids))

        train_ids = dm.get_split_sequence_ids("train")
        assert set(train_ids.tolist()).issubset(set(dm.train_sequence_ids))

        val_ids = dm.get_split_sequence_ids("val")
        assert set(val_ids.tolist()).issubset(set(dm.val_sequence_ids))

    def test_order_matches_subset_indices(self, split_data):
        """Returned sequence_ids are in the same order as the Subset indices."""
        dm = self._make_dm(split_data)

        for split_name in ("train", "val", "test"):
            dataset = getattr(dm, f"{split_name}_dataset")
            seq_ids = dm.get_split_sequence_ids(split_name)
            expected = dm.sequence_ids[dataset.indices]
            np.testing.assert_array_equal(seq_ids, expected)

    def test_invalid_split_name_raises(self, split_data):
        """Invalid split name raises ValueError."""
        dm = self._make_dm(split_data)
        with pytest.raises(ValueError):
            dm.get_split_sequence_ids("invalid")


class TestSaveSplitAssignment:
    """Tests for Phase 1d: persist split assignment as JSON."""

    def _make_dm(self, data, split_seed=0):
        dm = TimeSeriesDataModule(
            feature_path=data["features_path"],
            target_path=data["targets_path"],
            sequence_ids_path=data["sequence_ids_path"],
            batch_size=4,
            split_seed=split_seed,
        )
        dm.setup()
        return dm

    def test_save_creates_json_file(self, split_data, tmp_path):
        """save_split_assignment creates a JSON file at the given path."""
        dm = self._make_dm(split_data)
        out_path = tmp_path / "split.json"
        dm.save_split_assignment(str(out_path))
        assert out_path.exists()

    def test_json_contains_required_keys(self, split_data, tmp_path):
        """Saved JSON contains train/val/test sequence IDs and metadata."""
        dm = self._make_dm(split_data)
        out_path = tmp_path / "split.json"
        dm.save_split_assignment(str(out_path))

        with open(out_path) as f:
            data = json.load(f)

        assert "train_sequence_ids" in data
        assert "val_sequence_ids" in data
        assert "test_sequence_ids" in data
        assert "split_seed" in data

    def test_json_values_match_split(self, split_data, tmp_path):
        """Saved JSON values match the actual split assignment."""
        dm = self._make_dm(split_data)
        out_path = tmp_path / "split.json"
        dm.save_split_assignment(str(out_path))

        with open(out_path) as f:
            data = json.load(f)

        assert data["train_sequence_ids"] == dm.train_sequence_ids
        assert data["val_sequence_ids"] == dm.val_sequence_ids
        assert data["test_sequence_ids"] == dm.test_sequence_ids
        assert data["split_seed"] == dm.split_seed

    def test_json_contains_sample_counts(self, split_data, tmp_path):
        """Saved JSON includes sample counts per split."""
        dm = self._make_dm(split_data)
        out_path = tmp_path / "split.json"
        dm.save_split_assignment(str(out_path))

        with open(out_path) as f:
            data = json.load(f)

        assert data["n_train_samples"] == len(dm.train_dataset)
        assert data["n_val_samples"] == len(dm.val_dataset)
        assert data["n_test_samples"] == len(dm.test_dataset)
