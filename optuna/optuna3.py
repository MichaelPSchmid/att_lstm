import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from model.LSTM_attention import LSTMAttentionModel
from data_module import TimeSeriesDataModule
import torch
import gc

# Set PyTorch optimization strategy to reduce CPU computation burden
torch.set_float32_matmul_precision('medium')
torch.set_num_threads(4)

def objective(trial):
    """ Objective function for Optuna hyperparameter tuning """

    # Sample hyperparameters
    hidden_size = trial.suggest_int("hidden_size", 16, 64, step=16)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    # Set a fixed batch size
    batch_size = 16  

    # Dataset file paths
    feature_path = "/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF_NewFeatures/feature_50_1_1_sF_nF.pkl"
    target_path = "/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF_NewFeatures/target_50_1_1_sF_nF.pkl"

    # Configure data loading parameters to reduce memory usage
    data_module = TimeSeriesDataModule(feature_path, target_path, batch_size=batch_size)

    # Initialize model
    input_size = 7
    output_size = 1
    model = LSTMAttentionModel(input_size, hidden_size, num_layers, output_size, lr).to("cuda" if torch.cuda.is_available() else "cpu")

    # Early stopping to reduce unnecessary computation
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Set up the trainer
    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop_callback],
        logger=False,
        enable_checkpointing=False,
        precision="16-mixed" if torch.cuda.is_available() else 32,  # Use mixed precision to reduce GPU memory usage
    )

    try:
        # Train the model
        trainer.fit(model, data_module)

        # Validate the model
        val_result = trainer.validate(model, dataloaders=data_module.val_dataloader(), verbose=False)
        val_loss = val_result[0]["val_loss"]

    except Exception as e:
        print(f"Trial failed: {e}")
        val_loss = float("inf")  # Assign high loss if training fails to prevent affecting overall search

    finally:
        # Cleanup to prevent memory leaks
        del model
        del data_module
        torch.cuda.empty_cache()
        gc.collect()

    return val_loss

if __name__ == "__main__":
    # Use SQLite storage to save Optuna results and avoid excessive RAM usage
    study = optuna.create_study(direction="minimize", storage="sqlite:///optuna.db", study_name="lstm_tuning", load_if_exists=True)
    
    # Run hyperparameter tuning with a single-threaded process to reduce memory consumption
    study.optimize(objective, n_trials=15, n_jobs=1)  

    print("Best hyperparameters found:", study.best_params)
