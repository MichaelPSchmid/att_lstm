import json
from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch
import pickle
import numpy as np


def load_data_file(file_path: str) -> np.ndarray:
    """
    Load data from either .npy or .pkl file.

    Automatically detects format based on file extension or tries numpy first.
    For large datasets, prefers .npy files as they are more memory-efficient.

    Args:
        file_path: Path to the data file (.npy or .pkl)

    Returns:
        numpy array with the data
    """
    path = Path(file_path)

    # Check for .npy version first (more efficient)
    npy_path = path.with_suffix('.npy')
    if npy_path.exists():
        print(f"Loading numpy file: {npy_path}")
        return np.load(npy_path).astype(np.float32)

    # Fall back to pickle
    if path.exists():
        print(f"Loading pickle file: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, list):
                data = np.array(data, dtype=np.float32)
            return data.astype(np.float32)

    raise FileNotFoundError(f"No data file found at {file_path} or {npy_path}")


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, feature_path, target_path, sequence_ids_path, batch_size=32,
                 split_seed=0):
        super(TimeSeriesDataModule, self).__init__()
        self.feature_path = feature_path
        self.target_path = target_path
        self.sequence_ids_path = sequence_ids_path
        self.batch_size = batch_size
        self.split_seed = split_seed
        self.sequence_ids = None
        self.train_sequence_ids = None
        self.val_sequence_ids = None
        self.test_sequence_ids = None

    def prepare_data(self):
        """Load feature and target data from numpy or pickle files."""
        # Load features
        data = load_data_file(self.feature_path)
        self.X = torch.tensor(data, dtype=torch.float32)

        # Load targets - keep shape [N, 1] to match model output
        data = load_data_file(self.target_path)
        self.Y = torch.tensor(data, dtype=torch.float32)
        if self.Y.dim() == 1:
            self.Y = self.Y.unsqueeze(-1)

        print(f"Loaded features: {self.X.shape}")
        print(f"Loaded targets: {self.Y.shape}")

        # Load sequence_ids
        seq_ids_path = Path(self.sequence_ids_path)
        if not seq_ids_path.exists():
            raise FileNotFoundError(
                f"Sequence IDs file not found: {self.sequence_ids_path}"
            )
        with open(seq_ids_path, "rb") as f:
            raw_ids = pickle.load(f)
        self.sequence_ids = np.array(raw_ids, dtype=np.int64)
        print(f"Loaded sequence_ids: {len(self.sequence_ids)} "
              f"({len(np.unique(self.sequence_ids))} unique sequences)")

    def setup(self, stage=None):

        self.prepare_data()

        dataset = TensorDataset(self.X, self.Y)

        # Sequence-level split: split unique sequence IDs, then derive sample indices
        unique_seq_ids = np.unique(self.sequence_ids)
        n_sequences = len(unique_seq_ids)

        # Shuffle sequence IDs with dedicated split RNG (independent of training seed)
        rng = np.random.RandomState(self.split_seed)
        shuffled_ids = unique_seq_ids.copy()
        rng.shuffle(shuffled_ids)

        # Split on sequence level: 70/20/10
        n_train = int(0.7 * n_sequences)
        n_val = int(0.2 * n_sequences)

        self.train_sequence_ids = sorted(shuffled_ids[:n_train].tolist())
        self.val_sequence_ids = sorted(shuffled_ids[n_train:n_train + n_val].tolist())
        self.test_sequence_ids = sorted(shuffled_ids[n_train + n_val:].tolist())

        # Derive sample indices from sequence assignments
        train_seq_set = set(self.train_sequence_ids)
        val_seq_set = set(self.val_sequence_ids)
        test_seq_set = set(self.test_sequence_ids)

        train_indices = np.where(np.isin(self.sequence_ids, list(train_seq_set)))[0].tolist()
        val_indices = np.where(np.isin(self.sequence_ids, list(val_seq_set)))[0].tolist()
        test_indices = np.where(np.isin(self.sequence_ids, list(test_seq_set)))[0].tolist()

        self.train_dataset = torch.utils.data.Subset(dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(dataset, test_indices)

        print(f"Sequence-level split (seed={self.split_seed}): "
              f"{len(self.train_sequence_ids)} train / "
              f"{len(self.val_sequence_ids)} val / "
              f"{len(self.test_sequence_ids)} test sequences")
        print(f"Sample counts: {len(train_indices)} train / "
              f"{len(val_indices)} val / {len(test_indices)} test")


    def save_split_assignment(self, path: str) -> None:
        """Save the split assignment to a JSON file.

        Args:
            path: Output file path.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        assignment = {
            "split_seed": self.split_seed,
            "train_sequence_ids": self.train_sequence_ids,
            "val_sequence_ids": self.val_sequence_ids,
            "test_sequence_ids": self.test_sequence_ids,
            "n_train_samples": len(self.train_dataset),
            "n_val_samples": len(self.val_dataset),
            "n_test_samples": len(self.test_dataset),
            "n_train_sequences": len(self.train_sequence_ids),
            "n_val_sequences": len(self.val_sequence_ids),
            "n_test_sequences": len(self.test_sequence_ids),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(assignment, f, indent=2)
        print(f"Split assignment saved to: {out_path}")

    def get_split_sequence_ids(self, split: str) -> np.ndarray:
        """Get per-sample sequence_ids for a given split.

        Args:
            split: One of "train", "val", "test".

        Returns:
            Array of sequence_ids aligned with the split's sample order.
        """
        datasets = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset,
        }
        if split not in datasets:
            raise ValueError(f"Invalid split '{split}', must be one of {list(datasets)}")
        return self.sequence_ids[datasets[split].indices]

    def train_dataloader(self):
        print(f"Number of training samples: {len(self.train_dataset)}")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        print(f"Number of validation samples: {len(self.val_dataset)}")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        print(f"Number of test samples: {len(self.test_dataset)}")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
