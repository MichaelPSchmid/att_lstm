"""
Callback for saving attention weights during training.

This callback collects attention weights during validation and test phases
and saves them as numpy files for later analysis.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)


class AttentionSaveCallback(pl.Callback):
    """
    Callback to save attention weights during training.

    Collects attention weights from models that support `forward(x, return_attention=True)`.
    Saves averaged weights per epoch and final test weights.

    Args:
        output_dir: Directory to save attention weights
        save_per_epoch: Whether to save weights after each validation epoch
        save_csv: Whether to also save as CSV (requires pandas)
    """

    def __init__(
        self,
        output_dir: str = "attention_weights",
        save_per_epoch: bool = True,
        save_csv: bool = True,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.save_per_epoch = save_per_epoch
        self.save_csv = save_csv

        # Accumulators for attention weights
        self._val_weights_sum: Optional[torch.Tensor] = None
        self._val_count: int = 0
        self._test_weights_sum: Optional[torch.Tensor] = None
        self._test_count: int = 0

        # Storage for all epochs
        self._all_epochs_data: Dict[str, np.ndarray] = {}

    def _has_attention(self, model: pl.LightningModule) -> bool:
        """Check if model supports attention weight extraction."""
        # Check if forward accepts return_attention parameter
        import inspect
        sig = inspect.signature(model.forward)
        return "return_attention" in sig.parameters

    def _extract_attention(
        self, model: pl.LightningModule, x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Extract attention weights from model."""
        try:
            output, attention_weights = model.forward(x, return_attention=True)
            return attention_weights
        except Exception as e:
            logger.warning(f"Failed to extract attention weights: {e}")
            return None

    def _reduce_attention_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Reduce attention weights to a 1D vector per sample for averaging.

        Different attention mechanisms have different output shapes:
        - Simple attention: (batch, seq_len)
        - Scaled dot-product: (batch, seq_len) or (batch, seq_len, seq_len)
        - Additive attention: (batch, seq_len, seq_len)

        For matrix attention (seq_len x seq_len), we take the last row
        (attention from last timestep to all others) as this is most
        relevant for prediction.
        """
        if weights.dim() == 2:
            # Already (batch, seq_len)
            return weights
        elif weights.dim() == 3:
            # (batch, seq_len, seq_len) -> take last row
            return weights[:, -1, :]
        else:
            logger.warning(f"Unexpected attention shape: {weights.shape}")
            return weights.view(weights.size(0), -1)

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Called when fit or test begins."""
        if not self._has_attention(pl_module):
            logger.warning(
                f"Model {pl_module.__class__.__name__} does not support "
                "return_attention=True. Attention saving will be skipped."
            )
            return

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Attention weights will be saved to: {self.output_dir}")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect attention weights after each validation batch."""
        if not self._has_attention(pl_module):
            return

        x, _ = batch
        attention_weights = self._extract_attention(pl_module, x)

        if attention_weights is None:
            return

        # Reduce to 1D per sample and accumulate
        reduced = self._reduce_attention_weights(attention_weights)
        batch_sum = reduced.detach().cpu().sum(dim=0)

        if self._val_weights_sum is None:
            self._val_weights_sum = batch_sum
        else:
            self._val_weights_sum += batch_sum

        self._val_count += reduced.size(0)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Save averaged attention weights at end of validation epoch."""
        if self._val_weights_sum is None or self._val_count == 0:
            return

        # Calculate average
        avg_weights = (self._val_weights_sum / self._val_count).numpy()
        epoch = trainer.current_epoch

        # Store for later
        self._all_epochs_data[f"epoch_{epoch}"] = avg_weights

        if self.save_per_epoch:
            # Save individual epoch file
            epoch_path = self.output_dir / f"attention_epoch_{epoch:03d}.npy"
            np.save(epoch_path, avg_weights)
            logger.info(f"Saved attention weights for epoch {epoch}")

        # Reset accumulators
        self._val_weights_sum = None
        self._val_count = 0

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect attention weights after each test batch."""
        if not self._has_attention(pl_module):
            return

        x, _ = batch
        attention_weights = self._extract_attention(pl_module, x)

        if attention_weights is None:
            return

        # Reduce to 1D per sample and accumulate
        reduced = self._reduce_attention_weights(attention_weights)
        batch_sum = reduced.detach().cpu().sum(dim=0)

        if self._test_weights_sum is None:
            self._test_weights_sum = batch_sum
        else:
            self._test_weights_sum += batch_sum

        self._test_count += reduced.size(0)

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Save averaged attention weights at end of test phase."""
        if self._test_weights_sum is None or self._test_count == 0:
            return

        # Calculate average
        avg_weights = (self._test_weights_sum / self._test_count).numpy()

        # Save test weights
        test_path = self.output_dir / "attention_test.npy"
        np.save(test_path, avg_weights)
        logger.info(f"Saved test attention weights to {test_path}")

        # Also save to all epochs data
        self._all_epochs_data["test"] = avg_weights

        # Reset accumulators
        self._test_weights_sum = None
        self._test_count = 0

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save all collected attention weights at end of training."""
        if not self._all_epochs_data:
            return

        # Add time steps index
        if self._all_epochs_data:
            first_key = next(iter(self._all_epochs_data))
            seq_len = len(self._all_epochs_data[first_key])
            self._all_epochs_data["time_steps"] = np.arange(seq_len)

        # Save as NPZ
        npz_path = self.output_dir / "attention_all_epochs.npz"
        np.savez(npz_path, **self._all_epochs_data)
        logger.info(f"Saved all attention weights to {npz_path}")

        # Optionally save as CSV
        if self.save_csv:
            self._save_csv()

    def _save_csv(self) -> None:
        """Save attention weights as CSV for easy viewing."""
        try:
            import pandas as pd

            # Build DataFrame
            data = {"time_step": self._all_epochs_data.get("time_steps", [])}

            for key, values in sorted(self._all_epochs_data.items()):
                if key != "time_steps":
                    data[key] = values

            df = pd.DataFrame(data)
            csv_path = self.output_dir / "attention_all_epochs.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved attention weights to CSV: {csv_path}")

        except ImportError:
            logger.warning("pandas not available, skipping CSV export")
        except Exception as e:
            logger.warning(f"Failed to save CSV: {e}")
