from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
import torch
import pickle
import numpy as np

class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, feature_path, target_path, batch_size=32, random_seed=42):
        super(TimeSeriesDataModule, self).__init__()
        self.feature_path = feature_path
        self.target_path = target_path
        self.batch_size = batch_size
        self.random_seed = random_seed 
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        
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

    def setup(self, stage=None):
        
        if self.train_dataset is None:  
            dataset = TensorDataset(self.X, self.Y)
            dataset_size = len(dataset)
            # print(f"self.X shape: {self.X.shape}")
            # print(f"self.Y shape: {self.Y.shape}")

            
            train_size = int(0.7 * dataset_size)
            val_size = int(0.2 * dataset_size)
            test_size = dataset_size - train_size - val_size 

        
            generator = torch.Generator().manual_seed(self.random_seed)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, [train_size, val_size, test_size], generator=generator
            )

    def train_dataloader(self):
        print(f"Number of training samples: {len(self.train_dataset)}")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        print(f"Number of validation samples: {len(self.val_dataset)}")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        print(f"Number of test samples: {len(self.test_dataset)}")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
