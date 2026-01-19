import pytorch_lightning as pl
from model.attention_visualization.additive_attention_2 import LSTMAdditiveAttentionModel
from data_module import TimeSeriesDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import numpy as np
import json

from config import FEATURE_PATH, TARGET_PATH, LIGHTNING_LOGS_DIR, ATTENTION_VIS_DIR

class AttentionWeightSaveCallback(pl.Callback):
    def __init__(self, output_dir="attention_weights", filename="timestep_importance.npz"):
        super().__init__()
        self.output_dir = output_dir
        self.filename = filename
        self.all_epochs_data = {}
        os.makedirs(output_dir, exist_ok=True)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Check if the model has saved attention weights
        if hasattr(pl_module, 'epoch_attention_weights') and pl_module.epoch_attention_weights is not None:
            # Get weights
            attn_weights = pl_module.epoch_attention_weights.cpu().numpy()
            
            # Ensure weights are a flattened 1D array
            if len(attn_weights.shape) > 1:
                attn_weights = attn_weights.flatten()
            
            # Normalize the weights for better visualization
            normalized_weights = attn_weights / np.sum(attn_weights)
            
            # Create time step index
            time_steps = np.arange(len(attn_weights))
            
            # Save current epoch data
            epoch_num = trainer.current_epoch
            self.all_epochs_data[f'epoch_{epoch_num}'] = attn_weights
            self.all_epochs_data[f'epoch_{epoch_num}_normalized'] = normalized_weights
            
            # Save time step index for the first epoch
            if 'time_steps' not in self.all_epochs_data:
                self.all_epochs_data['time_steps'] = time_steps
            
            # Save all accumulated data
            save_path = os.path.join(self.output_dir, self.filename)
            np.savez(save_path, **self.all_epochs_data)
            
            print(f"Updated cumulative timestep importance data with epoch {epoch_num}")
    
    def on_fit_end(self, trainer, pl_module):
        # Save accumulated data in a more readable CSV format at the end of training
        save_path_csv = os.path.join(self.output_dir, "all_epochs_timestep_importance.csv")
        
        try:
            import pandas as pd
            
            # Create DataFrame
            data_dict = {}
            data_dict['time_step'] = self.all_epochs_data['time_steps']
            
            # Add weights for each epoch
            for key in sorted(self.all_epochs_data.keys()):
                if key != 'time_steps' and not key.endswith('_normalized'):
                    data_dict[key] = self.all_epochs_data[key]
            
            df = pd.DataFrame(data_dict)
            df.to_csv(save_path_csv, index=False)
            
            # Also save normalized weights
            norm_data_dict = {}
            norm_data_dict['time_step'] = self.all_epochs_data['time_steps']
            
            for key in sorted(self.all_epochs_data.keys()):
                if key.endswith('_normalized'):
                    norm_data_dict[key] = self.all_epochs_data[key]
            
            if len(norm_data_dict) > 1:  # If we have more than just the time_step column
                norm_df = pd.DataFrame(norm_data_dict)
                norm_save_path = os.path.join(self.output_dir, "all_epochs_normalized_importance.csv")
                norm_df.to_csv(norm_save_path, index=False)
                print(f"Saved normalized timestep importance to CSV: {norm_save_path}")
            
            print(f"Saved all epochs timestep importance to CSV: {save_path_csv}")
        except ImportError:
            print("pandas not available, skipping CSV export")

# Set random seed to ensure experiment reproducibility
pl.seed_everything(3407)

# Enable Tensor Cores optimization
torch.set_float32_matmul_precision('medium')

# File paths (from config)
feature_path = FEATURE_PATH
target_path = TARGET_PATH

# Create attention visualization directory
attention_vis_dir = ATTENTION_VIS_DIR
attention_vis_dir.mkdir(parents=True, exist_ok=True)

data_module = TimeSeriesDataModule(str(feature_path), str(target_path), batch_size=32)

# Initialize model
model = LSTMAdditiveAttentionModel(
    input_size=5, hidden_size=128, num_layers=5, output_size=1, 
    lr=0.000382819
)

# Callbacks
early_stop_callback = EarlyStopping(
    monitor="val_loss", patience=5, mode="min"
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=3,
    mode="min",
    filename="LSTMAdditiveAttentionModel-{epoch:02d}-{val_loss:.4f}"
)

# Add attention visualization callback
attention_callback = AttentionWeightSaveCallback(output_dir=str(attention_vis_dir))

# Logger
logger = TensorBoardLogger(str(LIGHTNING_LOGS_DIR), name="LSTMAdditiveAttentionModel")

# Trainer setup with additional options
trainer = pl.Trainer(
    max_epochs=2,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    callbacks=[checkpoint_callback, early_stop_callback, attention_callback],
    logger=logger,
    enable_checkpointing=True,
    log_every_n_steps=50,
)

# Start training
trainer.fit(model, data_module)

# Test
trainer.test(model, dataloaders=data_module.test_dataloader())

# Save test set timestep importance
if hasattr(model, 'test_avg_attention_weights'):
    test_weights = model.test_avg_attention_weights.cpu().numpy()
    
    # Ensure weights are a flattened 1D array
    if len(test_weights.shape) > 1:
        test_weights = test_weights.flatten()
    
    # Calculate normalized weights
    normalized_weights = test_weights / np.sum(test_weights)
    
    # Save as NumPy compressed file (.npz)
    npz_path = attention_vis_dir / "test_timestep_importance.npz"
    np.savez(npz_path,
             test_importance=test_weights,
             test_normalized_importance=normalized_weights)
    print(f"Saved test timestep importance to {npz_path}")

    # Optional: Save as CSV for easy viewing
    try:
        import pandas as pd
        csv_path = attention_vis_dir / "test_timestep_importance.csv"
        df = pd.DataFrame({
            'time_step': np.arange(len(test_weights)),
            'importance': test_weights,
            'normalized_importance': normalized_weights
        })
        df.to_csv(csv_path, index=False)
        print(f"Saved test timestep importance to CSV: {csv_path}")
        
    except ImportError:
        print("pandas or matplotlib not available, skipping CSV export and visualization")
    
    print("Saved test set timestep importance")

