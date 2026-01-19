import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VariableDotAttention(nn.Module):
    def __init__(self):
        super(VariableDotAttention, self).__init__()

    def forward(self, x):
        """
        Feature-based (variable) attention mechanism.

        Args:
            x (tensor): Shape (batch, seq_len, input_size)

        Returns:
            tensor: Shape (batch, seq_len, input_size) after applying feature attention.
        """
        # Compute attention scores based on feature relationships (input_size dimension)
        e = torch.bmm(x.permute(0, 2, 1), x)  # (batch, input_size, seq_len) * (batch, seq_len, input_size) -> (batch, input_size, input_size)
        attention = F.softmax(e, dim=-1)  # Normalize attention scores (batch, input_size, input_size)
        
        # Apply attention weights to features
        out = torch.bmm(attention, x.permute(0, 2, 1))  # (batch, input_size, input_size) * (batch, input_size, seq_len) -> (batch, input_size, seq_len)
        
        # Restore original shape (batch, seq_len, input_size)
        return F.relu(out.permute(0, 2, 1))

class LSTMDotAttentionModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr=0.001):
        super(LSTMDotAttentionModel, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Apply feature-based attention instead of time-step attention
        self.attention = VariableDotAttention()

        # Fully connected layer
        self.fc = nn.Linear(input_size, output_size)

        # Loss function
        self.criterion = nn.MSELoss()
        self.lr = lr

        # Metrics accumulation
        self.rmse_sum = 0.0
        self.mape_sum = 0.0
        self.sum_abs_correct = 0.0
        self.total_samples = 0

    def forward(self, x):
        """
        Forward pass through LSTM and variable-based attention.

        Args:
            x (tensor): Shape (batch, seq_len, input_size)

        Returns:
            tensor: Final prediction (batch, output_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_output, (hn, _) = self.lstm(x)  # Output shape: (batch, seq_len, hidden_size)

        att_output = self.attention(x)  # Apply feature-based attention (batch, seq_len, input_size)

        output = self.fc(att_output[:, -1, :])  # Use last time step's feature for final prediction

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
        Validation step with RMSE, MAPE, and Absolute Error Accuracy.

        Args:
            batch (tuple): (X_batch, Y_batch)
            batch_idx (int): Batch index

        Returns:
            tensor: Validation loss
        """
        X_batch, Y_batch = batch
        outputs = self.forward(X_batch)
        loss = self.criterion(outputs, Y_batch)

        # RMSE Calculation
        rmse = torch.sqrt(torch.mean((outputs - Y_batch) ** 2))
        self.rmse_sum += rmse.item()

        # MAPE Calculation (avoiding division by zero)
        mape = torch.mean(torch.abs((outputs - Y_batch) / (Y_batch + 1e-8))) 
        self.mape_sum += mape.item()

        # Absolute Error Accuracy Calculation
        abs_threshold = 0.05  # Define the acceptable error range
        abs_correct = torch.abs(outputs - Y_batch) < abs_threshold
        self.sum_abs_correct += abs_correct.sum().item()
        self.total_samples += Y_batch.numel()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_rmse", rmse, prog_bar=True, on_epoch=True)
        self.log("val_mape", mape, prog_bar=True, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """
        Compute and log average validation metrics at the end of an epoch.
        """
        avg_abs_accuracy = self.sum_abs_correct / self.total_samples
        avg_rmse = self.rmse_sum / self.total_samples
        avg_mape = self.mape_sum / self.total_samples

        self.log("avg_val_abs_accuracy", avg_abs_accuracy, prog_bar=True)
        self.log("avg_val_rmse", avg_rmse, prog_bar=True)
        self.log("avg_val_mape", avg_mape, prog_bar=True)

        print(f"Validation Absolute Accuracy: {avg_abs_accuracy:.4f}, RMSE: {avg_rmse:.4f}, MAPE: {avg_mape:.4f}")

        # Reset accumulators
        self.sum_abs_correct = 0.0
        self.total_samples = 0
        self.rmse_sum = 0.0
        self.mape_sum = 0.0

    def test_step(self, batch, batch_idx):
        """
        Test step with RMSE, MAPE, and Absolute Error Accuracy.

        Args:
            batch (tuple): (X_batch, Y_batch)
            batch_idx (int): Batch index

        Returns:
            tensor: Test loss
        """
        X_batch, Y_batch = batch
        outputs = self.forward(X_batch)
        loss = self.criterion(outputs, Y_batch)

        rmse = torch.sqrt(torch.mean((outputs - Y_batch) ** 2))
        self.rmse_sum += rmse.item()

        mape = torch.mean(torch.abs((outputs - Y_batch) / (Y_batch + 1e-8))) 
        self.mape_sum += mape.item()

        abs_threshold = 0.05
        abs_correct = torch.abs(outputs - Y_batch) < abs_threshold
        self.sum_abs_correct += abs_correct.sum().item()
        self.total_samples += Y_batch.numel()

        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_rmse", rmse, prog_bar=True, on_epoch=True)
        self.log("test_mape", mape, prog_bar=True)

        return loss

    def on_test_epoch_end(self):
        """
        Compute and log average test metrics at the end of an epoch.
        """
        avg_abs_accuracy = self.sum_abs_correct / self.total_samples
        avg_rmse = self.rmse_sum / self.total_samples
        avg_mape = self.mape_sum / self.total_samples

        self.log("avg_test_abs_accuracy", avg_abs_accuracy, prog_bar=True)
        self.log("avg_test_rmse", avg_rmse, prog_bar=True)
        self.log("avg_test_mape", avg_mape, prog_bar=True)

        print(f"Test Absolute Accuracy: {avg_abs_accuracy:.4f}, RMSE: {avg_rmse:.4f}, MAPE: {avg_mape:.4f}")

        # Reset accumulators
        self.sum_abs_correct = 0.0
        self.total_samples = 0
        self.rmse_sum = 0.0
        self.mape_sum = 0.0

    def configure_optimizers(self):
        """
        Configure the optimizer (Adam with a given learning rate).
        """
        return optim.Adam(self.parameters(), lr=self.lr)
