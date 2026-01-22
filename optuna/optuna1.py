import optuna
import sys
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lstm_simple_attention import LSTMAttentionModel
from model.data_module import TimeSeriesDataModule
from config.settings import get_preprocessed_paths

def objective(trial):
    # Suggest hyperparameters for tuning
    hidden_size = trial.suggest_int("hidden_size", 32, 128, step=16)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)

    batch_size = 16

    # File paths (from config)
    paths = get_preprocessed_paths("HYUNDAI_SONATA_2020", window_size=15, predict_size=1, step_size=1, suffix="s")
    feature_path = str(paths["features"])
    target_path = str(paths["targets"])

    # Initialize the data module with the suggested batch size
    data_module = TimeSeriesDataModule(feature_path, target_path, batch_size=batch_size)

    # Initialize the model with suggested hyperparameters
    input_size = 5
    output_size = 1
    model = LSTMAttentionModel(input_size, hidden_size, num_layers, output_size, lr)

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )

    # Trainer setup
    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop_callback],
        logger=False,
        enable_checkpointing=False,
    )

    # Fit the model
    trainer.fit(model, data_module)

    # Evaluate the model on the validation set
    val_result = trainer.validate(model, dataloaders=data_module.val_dataloader(), verbose=False)
    val_loss = val_result[0]["val_loss"]

    return val_loss

if __name__ == "__main__":
    # Create a study and optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameters
    print("Best hyperparameters found:")
    print(study.best_params)
