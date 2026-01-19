import pytorch_lightning as pl
from model.attention_visualization.scaled_dot_product_attention_visualize import LSTMScaledDotAttentionModel
from data_module import TimeSeriesDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import os
import json

class AttentionWeightSaveCallback(pl.Callback):
    def __init__(self, output_dir="attention_weights", filename="all_epochs_attention_weights.npz"):
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
            
            # Create time step index
            time_steps = np.arange(len(attn_weights))
            
            # Save current epoch data
            epoch_num = trainer.current_epoch
            self.all_epochs_data[f'epoch_{epoch_num}'] = attn_weights
            
            # Save time step index for the first epoch
            if 'time_steps' not in self.all_epochs_data:
                self.all_epochs_data['time_steps'] = time_steps
            
            # Save all accumulated data
            save_path = os.path.join(self.output_dir, self.filename)
            np.savez(save_path, **self.all_epochs_data)
            
            print(f"Updated cumulative attention weights data with epoch {epoch_num}")
    
    def on_fit_end(self, trainer, pl_module):
        # Save accumulated data in a more readable CSV format at the end of training
        save_path_csv = os.path.join(self.output_dir, "all_epochs_attention_weights.csv")
        
        try:
            import pandas as pd
            
            # Create DataFrame
            data_dict = {}
            data_dict['time_step'] = self.all_epochs_data['time_steps']
            
            # Add weights for each epoch
            for key in sorted(self.all_epochs_data.keys()):
                if key != 'time_steps':
                    data_dict[key] = self.all_epochs_data[key]
            
            df = pd.DataFrame(data_dict)
            df.to_csv(save_path_csv, index=False)
            print(f"Saved all epochs attention weights to CSV: {save_path_csv}")
        except ImportError:
            print("pandas not available, skipping CSV export")

# Set random seed to ensure experiment reproducibility
pl.seed_everything(3407)

# Enable Tensor Cores optimization
torch.set_float32_matmul_precision('medium')

# File paths
feature_path = "/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF/feature_50_1_1_sF.pkl"
target_path = "/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF/target_50_1_1_sF.pkl"

# Create attention visualization directory
attention_vis_dir = "attention_visualization"
os.makedirs(attention_vis_dir, exist_ok=True)

data_module = TimeSeriesDataModule(feature_path, target_path, batch_size=32)

# Initialize model
model = LSTMScaledDotAttentionModel(
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
    filename="LSTMScaledDotAttentionModel-{epoch:02d}-{val_loss:.4f}"
)

# Add attention visualization callback
attention_callback = AttentionWeightSaveCallback(output_dir=attention_vis_dir)

# Logger
logger = TensorBoardLogger("lightning_logs", name="LSTMScaledDotAttentionModel")

# Trainer setup with additional options
trainer = pl.Trainer(
    max_epochs=3,
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

# Save test set attention weights
if hasattr(model, 'test_avg_attention_weights'):
    test_weights = model.test_avg_attention_weights.cpu().numpy()
    
    # Ensure weights are a flattened 1D array
    if len(test_weights.shape) > 1:
        test_weights = test_weights.flatten()
    
    # Save as NumPy compressed file (.npz)
    npz_path = os.path.join(attention_vis_dir, "test_attention_weights.npz")
    np.savez(npz_path, test_attention_weights=test_weights)
    print(f"Saved test attention weights to {npz_path}")
    
    # Optional: Save as CSV for easy viewing
    try:
        import pandas as pd
        csv_path = os.path.join(attention_vis_dir, "test_attention_weights.csv")
        pd.DataFrame(test_weights, columns=['attention_weight']).to_csv(csv_path, index=False)
        print(f"Saved test attention weights to CSV: {csv_path}")
    except ImportError:
        print("pandas not available, skipping CSV export")
    
    print("Saved test set attention weights")