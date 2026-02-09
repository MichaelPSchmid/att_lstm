"""
Training script for LSTM-Attention models.

Usage:
    python scripts/train_model.py --config config/model_configs/m1_small_baseline.yaml
    python scripts/train_model.py --config config/model_configs/m2_small_simple_attn.yaml
    python scripts/train_model.py --config config/model_configs/m3_small_additive.yaml --save-attention
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config.loader import load_config, create_model_from_config, print_config
from model.data_module import TimeSeriesDataModule
from config.settings import get_preprocessed_paths
from scripts.callbacks import AttentionSaveCallback


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LSTM-Attention model for steering torque prediction"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to model config file (e.g., config/model_configs/m1_small_baseline.yaml)"
    )
    parser.add_argument(
        "--base-config", "-b",
        type=str,
        default=None,
        help="Path to base config file (default: config/base_config.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without training"
    )
    parser.add_argument(
        "--save-attention",
        action="store_true",
        help="Save attention weights (overrides config, enables saving)"
    )
    parser.add_argument(
        "--no-save-attention",
        action="store_true",
        help="Disable attention weight saving (overrides config)"
    )
    parser.add_argument(
        "--attention-dir",
        type=str,
        default=None,
        help="Directory for attention weights (default: attention_weights/<model_name>)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config, args.base_config)

    # Print configuration
    print_config(config)

    if args.dry_run:
        print("Dry run - exiting without training")
        return

    # Set seed for reproducibility
    seed = config["training"]["seed"]
    pl.seed_everything(seed)
    print(f"Random seed set to: {seed}")

    # Enable Tensor Cores optimization
    torch.set_float32_matmul_precision('medium')

    # Get data paths
    data_config = config["data"]
    paths = get_preprocessed_paths(
        vehicle=data_config["vehicle"],
        window_size=data_config["window_size"],
        predict_size=data_config["predict_size"],
        step_size=data_config["step_size"],
        suffix="sF",
        variant=data_config["variant"]
    )

    # Create data module
    data_module = TimeSeriesDataModule(
        feature_path=str(paths["features"]),
        target_path=str(paths["targets"]),
        batch_size=config["training"]["batch_size"]
    )

    # Create model
    model = create_model_from_config(config)
    model_name = config["model"]["name"]

    print(f"\nModel: {model_name}")
    print(f"Type: {config['model']['type']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup callbacks
    callbacks = []

    # Early stopping
    es_config = config["training"]["early_stopping"]
    if es_config["enabled"]:
        early_stop_callback = EarlyStopping(
            monitor=es_config["monitor"],
            patience=es_config["patience"],
            mode=es_config["mode"]
        )
        callbacks.append(early_stop_callback)

    # Checkpointing
    ckpt_config = config["training"]["checkpoint"]
    if ckpt_config["enabled"]:
        checkpoint_callback = ModelCheckpoint(
            monitor=ckpt_config["monitor"],
            save_top_k=ckpt_config["save_top_k"],
            mode=ckpt_config["mode"],
            filename=f"{model_name}" + "-{epoch:02d}-{val_loss:.4f}"
        )
        callbacks.append(checkpoint_callback)

    # Attention saving (config or CLI flag)
    attention_config = config.get("attention", {})
    save_attention = attention_config.get("enabled", False)

    # CLI flags override config
    if args.save_attention:
        save_attention = True
    if args.no_save_attention:
        save_attention = False

    if save_attention:
        attention_dir = args.attention_dir
        if attention_dir is None:
            base_dir = attention_config.get("output_dir", "attention_weights")
            attention_dir = Path(base_dir) / model_name
        attention_callback = AttentionSaveCallback(
            output_dir=str(attention_dir),
            save_per_epoch=attention_config.get("save_per_epoch", True),
            save_csv=attention_config.get("save_csv", True),
        )
        callbacks.append(attention_callback)
        print(f"Attention weights will be saved to: {attention_dir}")

    # Logger
    logger = TensorBoardLogger(
        save_dir=config["output"]["logs_dir"],
        name=model_name
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["training"]["accelerator"],
        devices=config["training"]["devices"],
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=ckpt_config["enabled"],
        log_every_n_steps=config["training"]["log_every_n_steps"],
    )

    # Train
    print(f"\nStarting training...")
    print(f"Data variant: {data_config['variant']}")
    print(f"Max epochs: {config['training']['max_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print("-" * 60)

    if args.resume:
        trainer.fit(model, data_module, ckpt_path=args.resume)
    else:
        trainer.fit(model, data_module)

    # Test
    print("\nRunning test evaluation...")
    trainer.test(model, dataloaders=data_module.test_dataloader())

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {logger.log_dir}")


if __name__ == "__main__":
    main()
