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
    def __init__(self, feature_path, target_path, batch_size=32):
        super(TimeSeriesDataModule, self).__init__()
        self.feature_path = feature_path
        self.target_path = target_path
        self.batch_size = batch_size

    def prepare_data(self):
        """Load feature and target data from numpy or pickle files."""
        # Load features
        data = load_data_file(self.feature_path)
        self.X = torch.tensor(data, dtype=torch.float32)

        # Load targets
        data = load_data_file(self.target_path)
        self.Y = torch.tensor(data, dtype=torch.float32).squeeze(-1)

        print(f"Loaded features: {self.X.shape}")
        print(f"Loaded targets: {self.Y.shape}")

    def setup(self, stage=None):

        self.prepare_data()

        dataset = TensorDataset(self.X, self.Y)
        dataset_size = len(dataset)

        # Calculate sizes for training, validation, and test sets
        train_size = int(0.7 * dataset_size)
        val_size = int(0.2 * dataset_size)
        test_size = dataset_size - train_size - val_size  # Remaining for test set

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        # # Segment the data sequentially
        # self.train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
        # self.val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
        # self.test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, dataset_size))

        # self.train_indices = range(0, train_size)
        # self.val_indices = range(train_size, train_size + val_size)
        # self.test_indices = range(train_size + val_size, dataset_size)

        # print(f"Train indices: {self.train_indices}")
        # print(f"Validation indices: {self.val_indices}")
        # print(f"Test indices: {self.test_indices}")


    def train_dataloader(self):
        print(f"Number of training samples: {len(self.train_dataset)}")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15, pin_memory=True)

    def val_dataloader(self):
        print(f"Number of validation samples: {len(self.val_dataset)}")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15, pin_memory=True)

    def test_dataloader(self):
        print(f"Number of test samples: {len(self.test_dataset)}")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15, pin_memory=True)