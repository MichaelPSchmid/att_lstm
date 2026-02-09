"""
MLP Baseline for Steering Torque Prediction.

Two variants:
- MLP-Last: Uses only the last timestep (5 features)
- MLP-Flat: Flattens all timesteps (50 * 5 = 250 features)
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim


class MLPModel(pl.LightningModule):
    """
    MLP Baseline model for comparison with LSTM-based models.

    Args:
        input_size: Number of input features per timestep (default: 5)
        hidden_sizes: List of hidden layer sizes (e.g., [64, 64])
        output_size: Output dimension (default: 1)
        lr: Learning rate
        dropout: Dropout rate between layers
        use_last_only: If True, use only last timestep (MLP-Last).
                       If False, flatten all timesteps (MLP-Flat).
        seq_len: Sequence length for MLP-Flat variant (default: 50)
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_sizes: list = None,
        output_size: int = 1,
        lr: float = 0.001,
        dropout: float = 0.0,
        use_last_only: bool = True,
        seq_len: int = 50,
    ):
        super(MLPModel, self).__init__()
        self.save_hyperparameters()

        # Store configuration
        self.use_last_only = use_last_only
        self.seq_len = seq_len
        self.lr = lr

        # Default hidden sizes if not provided
        if hidden_sizes is None:
            hidden_sizes = [64, 64] if use_last_only else [128, 64]

        # Compute actual input dimension
        if use_last_only:
            actual_input_size = input_size  # Just last timestep
        else:
            actual_input_size = input_size * seq_len  # Flatten all

        # Build MLP layers
        layers = []
        prev_size = actual_input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0 and i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.mlp = nn.Sequential(*layers)

        # Loss function
        self.criterion = nn.MSELoss()

        # Accumulation variables for metrics (same pattern as LSTM)
        self.sum_abs_correct = 0.0
        self.total_samples = 0
        self.sum_squared_error = 0.0
        self.sum_targets = 0.0
        self.sum_targets_squared = 0.0

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_size]

        Returns:
            Output tensor [batch, output_size]
        """
        if self.use_last_only:
            # MLP-Last: Use only the last timestep
            x = x[:, -1, :]  # [batch, input_size]
        else:
            # MLP-Flat: Flatten all timesteps
            x = x.view(x.size(0), -1)  # [batch, seq_len * input_size]

        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        outputs = self.forward(X_batch)
        loss = self.criterion(outputs, Y_batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        outputs = self.forward(X_batch)
        loss = self.criterion(outputs, Y_batch)

        # Accumulate for epoch-level metrics
        squared_errors = (outputs - Y_batch) ** 2
        self.sum_squared_error += squared_errors.sum().item()
        self.sum_targets += Y_batch.sum().item()
        self.sum_targets_squared += (Y_batch ** 2).sum().item()

        # Accuracy calculation
        abs_threshold = 0.05
        abs_correct = torch.abs(outputs - Y_batch) < abs_threshold
        self.sum_abs_correct += abs_correct.sum().item()
        self.total_samples += Y_batch.numel()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        n = self.total_samples
        if n == 0:
            return

        # Accuracy
        accuracy = self.sum_abs_correct / n

        # RMSE: sqrt(SS_res / n)
        rmse = (self.sum_squared_error / n) ** 0.5

        # R²: 1 - SS_res / SS_tot
        ss_res = self.sum_squared_error
        ss_tot = self.sum_targets_squared - (self.sum_targets ** 2) / n
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        self.log("val_accuracy", accuracy, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        self.log("val_r2", r2, prog_bar=True)

        print(f"Validation - Accuracy: {accuracy:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        # Reset accumulators
        self.sum_abs_correct = 0.0
        self.total_samples = 0
        self.sum_squared_error = 0.0
        self.sum_targets = 0.0
        self.sum_targets_squared = 0.0

    def test_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        outputs = self.forward(X_batch)
        loss = self.criterion(outputs, Y_batch)

        # Accumulate for epoch-level metrics
        squared_errors = (outputs - Y_batch) ** 2
        self.sum_squared_error += squared_errors.sum().item()
        self.sum_targets += Y_batch.sum().item()
        self.sum_targets_squared += (Y_batch ** 2).sum().item()

        # Accuracy calculation
        abs_threshold = 0.05
        abs_correct = torch.abs(outputs - Y_batch) < abs_threshold
        self.sum_abs_correct += abs_correct.sum().item()
        self.total_samples += Y_batch.numel()

        self.log("test_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        n = self.total_samples
        if n == 0:
            return

        # Accuracy
        accuracy = self.sum_abs_correct / n

        # RMSE: sqrt(SS_res / n)
        rmse = (self.sum_squared_error / n) ** 0.5

        # R²: 1 - SS_res / SS_tot
        ss_res = self.sum_squared_error
        ss_tot = self.sum_targets_squared - (self.sum_targets ** 2) / n
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        self.log("test_accuracy", accuracy, prog_bar=True)
        self.log("test_rmse", rmse, prog_bar=True)
        self.log("test_r2", r2, prog_bar=True)

        print(f"Test - Accuracy: {accuracy:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        # Reset accumulators
        self.sum_abs_correct = 0.0
        self.total_samples = 0
        self.sum_squared_error = 0.0
        self.sum_targets = 0.0
        self.sum_targets_squared = 0.0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
