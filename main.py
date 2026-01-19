import pytorch_lightning as pl
from model.LSTM import LSTMModel
from data_module import TimeSeriesDataModule
# from comparison.data_seed42 import TimeSeriesDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from config import FEATURE_PATH, TARGET_PATH, LIGHTNING_LOGS_DIR, EVALUATION_DIR

# Set random seed to ensure experiment reproducibility
pl.seed_everything(3407)

# Enable Tensor Cores optimization
torch.set_float32_matmul_precision('medium')

# File paths (from config.py)
feature_path = FEATURE_PATH
target_path = TARGET_PATH

data_module = TimeSeriesDataModule(str(feature_path), str(target_path), batch_size=32)

# # Manually call prepare_data to load X and Y
# data_module.prepare_data()

# # Call setup to prepare datasets and split indices
# data_module.setup()


# Initialize model with dropout and weight decay
model = LSTMModel(
    input_size=5, hidden_size=128, num_layers=5, output_size=1, 
    lr=0.000382819
)

# Checkpoint path - adjust this to your specific checkpoint filename
# checkpoint_path = LIGHTNING_LOGS_DIR / "LSTMAttentionModel/version_0/checkpoints/LSTMAttentionModel-epoch=26-val_loss=0.0014.ckpt"


# Callbacks
early_stop_callback = EarlyStopping(
    monitor="val_loss", patience=5, mode="min"
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=3,
    mode="min",
    filename="LSTMModel-{epoch:02d}-{val_loss:.4f}"
)


# Logger
logger = TensorBoardLogger(str(LIGHTNING_LOGS_DIR), name="LSTMModel")

# Trainer setup with additional options
trainer = pl.Trainer(
    max_epochs=80,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=logger,
    enable_checkpointing=True,
    log_every_n_steps=50,  # Reduce logging frequency
)

# Continue training from checkpoint
trainer.fit(model, data_module)

# Test the model after continued training
trainer.test(model, dataloaders=data_module.test_dataloader())

# predictions_path = EVALUATION_DIR / "test_predictions.pkl"
# targets_path = EVALUATION_DIR / "test_targets.pkl"

# # Switch to evaluation mode to ensure the model does not update gradients
# model.eval()

# # Lists to store predictions and target values
# predictions = []
# targets = []

# # Disable gradient computation to prevent unnecessary memory usage
# with torch.no_grad():
#     test_dataloader = data_module.test_dataloader()
#     for batch in test_dataloader:
#         # Retrieve input features and target values from the test set
#         X_batch, Y_batch = batch

#         # Generate predictions using the trained model
#         Y_pred = model(X_batch)

#         # Convert predictions and targets to numpy arrays and store them
#         predictions.extend(Y_pred.detach().cpu().numpy())  # Convert to numpy and store
#         targets.extend(Y_batch.detach().cpu().numpy())     # Convert to numpy and store

# # Print the first 10 predicted vs actual values for verification
# print("Predicted vs Actual:")
# for i in range(min(len(predictions), 10)):  # Display only the first 10 results
#     print(f"Prediction: {predictions[i]}, Actual: {targets[i]}")

# # Ensure the directories for saving results exist
# os.makedirs(os.path.dirname(predictions_path), exist_ok=True)  
# os.makedirs(os.path.dirname(targets_path), exist_ok=True)      

# # Save predictions to a pickle file
# with open(predictions_path, "wb") as f:
#     pickle.dump(predictions, f)
#     print(f"Predictions saved to: {predictions_path}")

# # Save actual target values to a pickle file
# with open(targets_path, "wb") as f:
#     pickle.dump(targets, f)
#     print(f"Targets saved to: {targets_path}")
