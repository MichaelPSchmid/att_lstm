import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMAttentionModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr=0.001):
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

        # Accumulation variables
        self.sum_abs_correct = 0.0
        self.total_samples = 0
        self.rmse_sum = 0.0
        self.mape_sum = 0.0
        self.r2_sum = 0.0

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_output, _ = self.lstm(x)
        attention_scores = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        output = self.fc(context_vector)

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

        # RMSE Calculation
        rmse = torch.sqrt(torch.mean((outputs - Y_batch) ** 2))
        self.rmse_sum += rmse.item()

        # MAPE Calculation (avoid division by zero)
        mape = torch.mean(torch.abs((outputs - Y_batch) / (Y_batch + 1e-8))) 
        self.mape_sum += mape.item()

        # R² Score Calculation
        ss_res = torch.sum((outputs - Y_batch) ** 2)
        ss_tot = torch.sum((Y_batch - torch.mean(Y_batch)) ** 2)
        r2_score = 1 - ss_res / (ss_tot + 1e-8)
        self.r2_sum += r2_score.item()

        # Accuracy calculation
        abs_threshold = 0.05
        abs_correct = torch.abs(outputs - Y_batch) < abs_threshold
        self.sum_abs_correct += abs_correct.sum().item()
        self.total_samples += Y_batch.numel()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_rmse", rmse, prog_bar=True, on_epoch=True)
        self.log("val_mape", mape, prog_bar=True, on_epoch=True)
        self.log("val_r2", r2_score, prog_bar=True, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        avg_abs_accuracy = self.sum_abs_correct / self.total_samples
        avg_rmse = self.rmse_sum / self.total_samples
        avg_mape = self.mape_sum / self.total_samples
        avg_r2 = self.r2_sum / self.total_samples

        self.log("avg_val_abs_accuracy", avg_abs_accuracy, prog_bar=True)
        self.log("avg_val_rmse", avg_rmse, prog_bar=True)
        self.log("avg_val_mape", avg_mape, prog_bar=True)
        self.log("avg_val_r2", avg_r2, prog_bar=True)

        print(f"Validation Absolute Accuracy: {avg_abs_accuracy:.4f}, RMSE: {avg_rmse:.4f}, MAPE: {avg_mape:.4f}, R²: {avg_r2:.4f}")

        self.sum_abs_correct = 0.0
        self.total_samples = 0
        self.rmse_sum = 0.0
        self.mape_sum = 0.0
        self.r2_sum = 0.0

    def test_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        outputs = self.forward(X_batch)
        loss = self.criterion(outputs, Y_batch)

        rmse = torch.sqrt(torch.mean((outputs - Y_batch) ** 2))
        self.rmse_sum += rmse.item()

        mape = torch.mean(torch.abs((outputs - Y_batch) / (Y_batch + 1e-8))) 
        self.mape_sum += mape.item()

        ss_res = torch.sum((outputs - Y_batch) ** 2)
        ss_tot = torch.sum((Y_batch - torch.mean(Y_batch)) ** 2)
        r2_score = 1 - ss_res / (ss_tot + 1e-8)
        self.r2_sum += r2_score.item()

        abs_threshold = 0.05
        abs_correct = torch.abs(outputs - Y_batch) < abs_threshold
        self.sum_abs_correct += abs_correct.sum().item()
        self.total_samples += Y_batch.numel()

        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_rmse", rmse, prog_bar=True, on_epoch=True)
        self.log("test_mape", mape, prog_bar=True, on_epoch=True)
        self.log("test_r2", r2_score, prog_bar=True, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        avg_abs_accuracy = self.sum_abs_correct / self.total_samples
        avg_rmse = self.rmse_sum / self.total_samples
        avg_mape = self.mape_sum / self.total_samples
        avg_r2 = self.r2_sum / self.total_samples

        self.log("avg_test_abs_accuracy", avg_abs_accuracy, prog_bar=True)
        self.log("avg_test_rmse", avg_rmse, prog_bar=True)
        self.log("avg_test_mape", avg_mape, prog_bar=True)
        self.log("avg_test_r2", avg_r2, prog_bar=True)

        print(f"Test Absolute Accuracy: {avg_abs_accuracy:.4f}, RMSE: {avg_rmse:.4f}, MAPE: {avg_mape:.4f}, R²: {avg_r2:.4f}")

        self.sum_abs_correct = 0.0
        self.total_samples = 0
        self.rmse_sum = 0.0
        self.mape_sum = 0.0
        self.r2_sum = 0.0

    def configure_optimizers(self):
        """
        Optimizer configuration
        """
        return optim.Adam(self.parameters(), lr=self.lr)