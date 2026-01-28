"""
Checkpoint discovery and management utilities.

Provides consistent checkpoint finding logic across all evaluation scripts.
The key design decision: search ALL versions for the best checkpoint,
not just the latest version.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .models import ModelConfig
from .paths import get_log_dir

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a discovered checkpoint."""

    path: Path
    val_loss: float  # Actual value from checkpoint file
    epoch: int
    version: int

    @property
    def name(self) -> str:
        """Return the checkpoint filename."""
        return self.path.name


def _extract_version_number(version_dir: Path) -> int:
    """Extract version number from directory name (version_X)."""
    try:
        return int(version_dir.name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _load_checkpoint_metadata(ckpt_path: Path) -> Optional[tuple]:
    """Load epoch and val_loss directly from checkpoint file.

    Args:
        ckpt_path: Path to the checkpoint file

    Returns:
        Tuple of (epoch, val_loss) or None if loading fails
    """
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Get epoch directly from checkpoint
        epoch = ckpt.get("epoch")
        if epoch is None:
            return None

        # Get val_loss from ModelCheckpoint callback state
        val_loss = None
        if "callbacks" in ckpt:
            for cb_key, cb_val in ckpt["callbacks"].items():
                if "ModelCheckpoint" in cb_key and isinstance(cb_val, dict):
                    val_loss = cb_val.get("current_score") or cb_val.get("best_model_score")
                    if val_loss is not None:
                        break

        if val_loss is None:
            return None

        return (int(epoch), float(val_loss))
    except Exception as e:
        logger.warning(f"Could not load metadata from {ckpt_path}: {e}")
        return None


def find_all_checkpoints(model: ModelConfig, variant: str) -> List[CheckpointInfo]:
    """Find all checkpoints for a model across all versions.

    Loads epoch and val_loss directly from checkpoint files to ensure
    accurate sorting (filename val_loss is rounded).

    Args:
        model: Model configuration
        variant: Either "dropout" or "no_dropout"

    Returns:
        List of CheckpointInfo objects, sorted by val_loss (best first)
    """
    log_dir = get_log_dir(model, variant)

    if not log_dir.exists():
        return []

    checkpoints = []

    # Search all version directories
    for version_dir in log_dir.glob("version_*"):
        version_num = _extract_version_number(version_dir)
        if version_num < 0:
            continue

        checkpoints_dir = version_dir / "checkpoints"
        if not checkpoints_dir.exists():
            continue

        for ckpt_path in checkpoints_dir.glob("*.ckpt"):
            metadata = _load_checkpoint_metadata(ckpt_path)
            if metadata is None:
                logger.warning(f"Skipping {ckpt_path}: could not load metadata")
                continue

            epoch, val_loss = metadata

            checkpoints.append(
                CheckpointInfo(
                    path=ckpt_path,
                    val_loss=val_loss,
                    epoch=epoch,
                    version=version_num,
                )
            )

    # Sort by val_loss (best first)
    checkpoints.sort(key=lambda x: x.val_loss)
    return checkpoints


def find_best_checkpoint(
    model: ModelConfig,
    variant: str,
    latest_version_only: bool = False,
) -> Optional[Path]:
    """Find the best checkpoint for a model (lowest val_loss).

    By default, searches ALL versions and returns the checkpoint with the
    lowest val_loss. This ensures the best model is always used, regardless
    of when it was trained.

    Args:
        model: Model configuration
        variant: Either "dropout" or "no_dropout"
        latest_version_only: If True, only search the latest version
                           (not recommended, best model might be in earlier version)

    Returns:
        Path to the best checkpoint, or None if not found
    """
    all_checkpoints = find_all_checkpoints(model, variant)

    if not all_checkpoints:
        return None

    if latest_version_only:
        # Filter to only the latest version
        max_version = max(ckpt.version for ckpt in all_checkpoints)
        all_checkpoints = [ckpt for ckpt in all_checkpoints if ckpt.version == max_version]

    if not all_checkpoints:
        return None

    # Already sorted by val_loss, return best
    return all_checkpoints[0].path
