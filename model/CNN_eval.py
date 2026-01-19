import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

class CNNModel(pl.LightningModule):
    def __init__(self, input_size, num_filters, kernel_size, output_size, lr=0.001):
        super(CNNModel, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.lr = lr

        # **CNN architecture**
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding="same")
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(num_filters)
        self.batchnorm2 = nn.BatchNorm1d(num_filters)

        # **Global Average Pooling (GAP)**
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # **Fully connected layer**
        self.fc = nn.Linear(num_filters, output_size)

        # Loss function
        self.criterion = nn.MSELoss()

        # **Evaluation metric accumulators**
        self.sum_abs_correct = 0.0
        self.total_samples = 0
        self.rmse_sum = 0.0
        self.mape_sum = 0.0

    def forward(self, x):
        # **Adjust input dimensions**: (batch, seq_len, input_size) → (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)

        # **Convolution layers + Batch Normalization + ReLU**
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))

        # **Global Average Pooling (GAP)**: Reduce time dimension to 1
        x = self.global_avg_pool(x).squeeze(2)  # (batch, num_filters, 1) → (batch, num_filters)

        # **Fully connected layer**
        output = self.fc(x)
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

        # **Compute RMSE**
        rmse = torch.sqrt(torch.mean((outputs - Y_batch) ** 2))
        self.rmse_sum += rmse.item()

        # **Compute MAPE (Avoid division by zero)**
        mape = torch.mean(torch.abs((outputs - Y_batch) / (Y_batch + 1e-8)))
        self.mape_sum += mape.item()

        # **Compute Absolute Accuracy**
        abs_threshold = 0.05
        abs_correct = torch.abs(outputs - Y_batch) < abs_threshold
        self.sum_abs_correct += abs_correct.sum().item()
        self.total_samples += Y_batch.numel()

        # **Log metrics**
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_rmse", rmse, prog_bar=True, on_epoch=True)
        self.log("val_mape", mape, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """ Compute average RMSE, MAPE, and Absolute Accuracy for the entire validation set """
        avg_abs_accuracy = self.sum_abs_correct / self.total_samples
        avg_rmse = self.rmse_sum / self.total_samples
        avg_mape = self.mape_sum / self.total_samples

        self.log("avg_val_abs_accuracy", avg_abs_accuracy, prog_bar=True)
        self.log("avg_val_rmse", avg_rmse, prog_bar=True)
        self.log("avg_val_mape", avg_mape, prog_bar=True)

        print(f"Validation Absolute Accuracy: {avg_abs_accuracy:.4f}, RMSE: {avg_rmse:.4f}, MAPE: {avg_mape:.4f}")

        # **Reset accumulators**
        self.sum_abs_correct = 0.0
        self.total_samples = 0
        self.rmse_sum = 0.0
        self.mape_sum = 0.0

    def test_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        outputs = self.forward(X_batch)
        loss = self.criterion(outputs, Y_batch)

        # **Compute RMSE**
        rmse = torch.sqrt(torch.mean((outputs - Y_batch) ** 2))
        self.rmse_sum += rmse.item()

        # **Compute MAPE**
        mape = torch.mean(torch.abs((outputs - Y_batch) / (Y_batch + 1e-8)))
        self.mape_sum += mape.item()

        # **Compute Absolute Accuracy**
        abs_threshold = 0.05
        abs_correct = torch.abs(outputs - Y_batch) < abs_threshold
        self.sum_abs_correct += abs_correct.sum().item()
        self.total_samples += Y_batch.numel()

        # **Log metrics**
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_rmse", rmse, prog_bar=True, on_epoch=True)
        self.log("test_mape", mape, prog_bar=True, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        """ Compute average RMSE, MAPE, and Absolute Accuracy for the entire test set """
        avg_abs_accuracy = self.sum_abs_correct / self.total_samples
        avg_rmse = self.rmse_sum / self.total_samples
        avg_mape = self.mape_sum / self.total_samples

        self.log("avg_test_abs_accuracy", avg_abs_accuracy, prog_bar=True)
        self.log("avg_test_rmse", avg_rmse, prog_bar=True)
        self.log("avg_test_mape", avg_mape, prog_bar=True)

        print(f"Test Absolute Accuracy: {avg_abs_accuracy:.4f}, RMSE: {avg_rmse:.4f}, MAPE: {avg_mape:.4f}")

        # **Reset accumulators**
        self.sum_abs_correct = 0.0
        self.total_samples = 0
        self.rmse_sum = 0.0
        self.mape_sum = 0.0

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

