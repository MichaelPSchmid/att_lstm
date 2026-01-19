import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from model.LSTM import LSTMModel
from data_module import TimeSeriesDataModule
import torch
import gc

# Enable Tensor Cores optimization
torch.set_float32_matmul_precision('medium')

def objective(trial):
    # Suggest hyperparameters
    hidden_size = trial.suggest_int("hidden_size", 32, 128, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)

    # Fixed batch_size
    batch_size = 32  # Fixed value

    # File paths/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/15_1_1_s/feature_15_1_1_s.pkl
    feature_path = "/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF/feature_50_1_1_sF.pkl"
    target_path = "/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF/target_50_1_1_sF.pkl"

    # Initialize data module with fixed batch size
    data_module = TimeSeriesDataModule(feature_path, target_path, batch_size=batch_size)

    # Initialize model with suggested hyperparameters
    input_size = 5
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, lr)

    # Early stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Trainer setup
    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop_callback],
        logger=False,
        enable_checkpointing=False,
    )

    try:
        # Training
        trainer.fit(model, data_module)
        # Validation
        val_result = trainer.validate(model, dataloaders=data_module.val_dataloader(), verbose=False)
        val_loss = val_result[0]["val_loss"]

    finally:
        # Cleanup
        del model
        del data_module
        torch.cuda.empty_cache()
        gc.collect()

    return val_loss

if __name__ == "__main__":
    # Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15, n_jobs=1)  # Single-threaded to avoid memory pressure
    print("Best hyperparameters found:", study.best_params)
