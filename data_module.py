from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch
import pickle
import numpy as np

class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, feature_path, target_path, batch_size=32):
        super(TimeSeriesDataModule, self).__init__()
        self.feature_path = feature_path
        self.target_path = target_path
        self.batch_size = batch_size

    def prepare_data(self):
        """ Load feature and target data from pickle files efficiently. """
        with open(self.feature_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, list):  
                data = np.array(data, dtype=np.float32)  # Convert list to numpy array
            self.X = torch.tensor(data, dtype=torch.float32)  # Convert to tensor

        with open(self.target_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, list):  
                data = np.array(data, dtype=np.float32)  # Convert list to numpy array
            self.Y = torch.tensor(data, dtype=torch.float32).squeeze(-1)  # Convert to tensor


        # Uncomment for debugging:
        # print(f"Feature shape (X): {self.X.shape}")  # Input data shape
        # print(f"Target shape (Y): {self.Y.shape}")   # Target data shape

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