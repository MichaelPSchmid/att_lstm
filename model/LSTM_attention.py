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

        # Accumulation variables
        self.sum_abs_correct = 0.0
        self.total_samples = 0

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

        # Calculate absolute error accuracy
        abs_threshold = 0.05
        abs_correct = torch.abs(outputs - Y_batch) < abs_threshold
        self.sum_abs_correct += abs_correct.sum().item()
        self.total_samples += Y_batch.numel()  # Accumulate total sample count

        return loss

    def on_validation_epoch_end(self):
        """
        Operations at the end of each validation epoch
        """
        # Calculate the average absolute error accuracy for the epoch
        avg_abs_accuracy = self.sum_abs_correct / self.total_samples

        # Log and print
        self.log("avg_val_abs_accuracy", avg_abs_accuracy, prog_bar=True)
        print(f"Validation Absolute Accuracy (epoch): {avg_abs_accuracy:.4f}")

        # Reset accumulation variables
        self.sum_abs_correct = 0.0
        self.total_samples = 0

    def test_step(self, batch, batch_idx):
        """
        Single test step
        """
        X_batch, Y_batch = batch
        outputs = self.forward(X_batch)
        loss = self.criterion(outputs, Y_batch)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)

        # Calculate absolute error accuracy
        abs_threshold = 0.05
        abs_correct = torch.abs(outputs - Y_batch) < abs_threshold
        self.sum_abs_correct += abs_correct.sum().item()
        self.total_samples += Y_batch.numel()  # Accumulate total sample count

        return loss

    def on_test_epoch_end(self):
        """
        Operations at the end of the test phase
        """
        # Calculate the average absolute error accuracy for the entire test phase
        avg_abs_accuracy = self.sum_abs_correct / self.total_samples

        # Log and print
        self.log("avg_test_abs_accuracy", avg_abs_accuracy, prog_bar=True)
        print(f"Test Absolute Accuracy (epoch): {avg_abs_accuracy:.4f}")

        # Reset accumulation variables
        self.sum_abs_correct = 0.0
        self.total_samples = 0

    def configure_optimizers(self):
        """
        Optimizer configuration
        """
        return optim.Adam(self.parameters(), lr=self.lr)