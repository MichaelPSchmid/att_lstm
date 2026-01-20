import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr=0.001):
        super(LSTMModel, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters

        # LSTM architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_output, (hn, _) = self.lstm(x, (h0, c0))
        output = self.fc(hn[-1])  # Using the last hidden state for prediction

        return output

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

        # Accuracy
        accuracy = self.sum_abs_correct / n

        # RMSE: sqrt(SS_res / n)
        rmse = (self.sum_squared_error / n) ** 0.5

        # R²: 1 - SS_res / SS_tot
        # SS_tot = sum((y - y_mean)²) = sum(y²) - (sum(y))² / n
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

