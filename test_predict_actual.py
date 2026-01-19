# Import the model class from model.py
from model.LSTM_attention import LSTMAttentionModel

# Import the data module class from data_module.py
from data_module import TimeSeriesDataModule

# Other necessary imports
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import torch

from config import get_preprocessed_paths

# File paths (from config)
paths = get_preprocessed_paths("HYUNDAI_SONATA_2020", window_size=5, predict_size=1, step_size=5, suffix="sF")
feature_path = str(paths["features"])
target_path = str(paths["targets"])

# Initialize the data module
data_module = TimeSeriesDataModule(feature_path, target_path, batch_size=32)

# Model parameters
input_size = 5
hidden_size = 64
num_layers = 5
output_size = 1

# Initialize the model
model = LSTMAttentionModel(input_size, hidden_size, num_layers, output_size, lr=0.001)

# Early stopping callback
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
)

# Initialize the trainer
trainer = pl.Trainer(
    max_epochs=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    callbacks=[early_stop_callback]
)

# Start training
trainer.fit(model, data_module)

# After training, print the first batch of validation data predictions
val_dataloader = data_module.val_dataloader()
for batch in val_dataloader:
    X_batch, Y_batch = batch
    break

model.eval()
with torch.no_grad():
    Y_pred = model(X_batch)

Y_pred = Y_pred.detach().cpu().numpy()
Y_batch = Y_batch.detach().cpu().numpy()

print("Predicted vs Actual:")
for i in range(len(Y_pred)):
    print(f"Prediction: {Y_pred[i].tolist()}, Actual: {Y_batch[i].tolist()}")

