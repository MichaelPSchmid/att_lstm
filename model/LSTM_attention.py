import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMAttentionModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr=0.001):
        """
        Parameters:
        - input_size: Number of input features
        - hidden_size: Dimension of the LSTM hidden state
        - num_layers: Number of LSTM layers
        - output_size: Number of time steps to predict (predict one or several future values of "steer")
        - lr: Learning rate
        """
        super(LSTMAttentionModel, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters

        # Model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)  # Calculate attention scores

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
        Forward pass:
        Input shape: (batch_size, sequence_length, input_size)
        Output shape: (batch_size, output_size)
        """
        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_output, _ = self.lstm(x)  # (batch_size, sequence_length, hidden_size)

        # Calculate attention scores
        attention_scores = self.attention(lstm_output)  # (batch_size, sequence_length, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, sequence_length, 1)

        # Weighted sum
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_size)

        # Fully connected layer
        output = self.fc(context_vector)  # (batch_size, output_size)

        return output

    def training_step(self, batch, batch_idx):
        """
        Single training step
        """
        X_batch, Y_batch = batch
        outputs = self.forward(X_batch)
        loss = self.criterion(outputs, Y_batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step
        """
        X_batch, Y_batch = batch
        outputs = self.forward(X_batch)
        loss = self.criterion(outputs, Y_batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

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

        return loss

    def on_validation_epoch_end(self):
        """
        Operations at the end of each validation epoch
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
        Single test step
        """
        X_batch, Y_batch = batch
        outputs = self.forward(X_batch)
        loss = self.criterion(outputs, Y_batch)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)

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

        return loss

    def on_test_epoch_end(self):
        """
        Operations at the end of the test phase
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
        Optimizer configuration
        """
        return optim.Adam(self.parameters(), lr=self.lr)