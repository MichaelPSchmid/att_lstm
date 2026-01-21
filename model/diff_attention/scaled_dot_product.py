import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        Applies Scaled Dot-Product Self-Attention.

        Args:
            x (tensor): Shape (batch, seq_len, hidden_size)

        Returns:
            tuple: (context_vector, attention_weights)
                - context_vector: Shape (batch, hidden_size) - weighted average context vector
                - attention_weights: Shape (batch, seq_len) - global attention weights
        """
        # Calculate scaled dot-product attention scores
        e = torch.bmm(x, x.permute(0, 2, 1))  # (b, s, h) * (b, h, s) -> (b, s, s)
        e = e / math.sqrt(self.hidden_size + 1e-8)  # Scale by sqrt(hidden_size)
        
        # Apply softmax normalization
        attention = F.softmax(e, dim=-1)  # (b, s, s)

        # Calculate global attention weights - average attention across all time steps
        global_attention = torch.mean(attention, dim=1)  # (b, s)
        
        # Use global attention weights to compute context vector
        context_vector = torch.bmm(global_attention.unsqueeze(1), x).squeeze(1)  # (b, h)
        
        return context_vector, global_attention


class LSTMScaleDotAttentionModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr=0.001, dropout=0.0):
        super(LSTMScaleDotAttentionModel, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Dropout between LSTM layers (only applied if num_layers > 1)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)

        # Scaled Dot-Product Attention
        self.attention = ScaledDotProductAttention(hidden_size)

        # Dropout before FC layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Loss function
        self.criterion = nn.MSELoss()
        self.lr = lr

        # Accumulation variables for metrics
        self.sum_abs_correct = 0.0
        self.total_samples = 0
        self.sum_squared_error = 0.0      # For RMSE and R² (SS_res)
        self.sum_targets = 0.0            # For R² (to compute mean)
        self.sum_targets_squared = 0.0    # For R² (SS_tot)

    def forward(self, x):
        """
        Forward pass through LSTM and self-attention.

        Args:
            x (tensor): Shape (batch, seq_len, input_size)

        Returns:
            tensor: Final prediction (batch, output_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_output, (hn, _) = self.lstm(x)  # LSTM output: (batch, seq_len, hidden_size)

        # Apply attention to get context vector and attention weights
        context_vector, attention_weights = self.attention(lstm_output)

        # Use context vector for prediction instead of the last time step
        output = self.fc(self.dropout(context_vector))

        return output

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (tuple): (X_batch, Y_batch)
            batch_idx (int): Batch index

        Returns:
            tensor: Training loss
        """
        X_batch, Y_batch = batch
        outputs = self.forward(X_batch)
        loss = self.criterion(outputs, Y_batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step with metrics accumulation.

        Args:
            batch (tuple): (X_batch, Y_batch)
            batch_idx (int): Batch index

        Returns:
            tensor: Validation loss
        """
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
        """
        Compute and log validation metrics at the end of an epoch.
        """
        n = self.total_samples

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
        """
        Test step with metrics accumulation.

        Args:
            batch (tuple): (X_batch, Y_batch)
            batch_idx (int): Batch index

        Returns:
            tensor: Test loss
        """
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
        """
        Compute and log test metrics at the end of the test phase.
        """
        n = self.total_samples

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
        """
        Configure optimizer with LR scheduler.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}