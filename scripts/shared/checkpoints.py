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
    val_loss: float  # From filename (rounded)
    epoch: int
    version: int
    actual_val_loss: Optional[float] = None  # From checkpoint file (precise)

    @property
    def name(self) -> str:
        """Return the checkpoint filename."""
        return self.path.name

    @property
    def best_val_loss(self) -> float:
        """Return the most accurate val_loss available."""
        return self.actual_val_loss if self.actual_val_loss is not None else self.val_loss


def parse_checkpoint_name(path: Path) -> Optional[tuple]:
    """Parse checkpoint filename to extract epoch.

    Expected format: ModelName-epoch=XX-val_loss=X.XXXX.ckpt

    Args:
        path: Path to the checkpoint file

    Returns:
        Tuple of (epoch, filename_val_loss) or None if parsing fails
    """
    name = path.stem
    try:
        # Extract epoch
        epoch_part = name.split("epoch=")[1].split("-")[0]
        epoch = int(epoch_part)

        # Extract val_loss from filename (rounded, used as fallback)
        val_loss_part = name.split("val_loss=")[1]
        val_loss = float(val_loss_part)

        return (epoch, val_loss)
    except (IndexError, ValueError):
        return None


def _extract_version_number(version_dir: Path) -> int:
    """Extract version number from directory name (version_X)."""
    try:
        return int(version_dir.name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _load_actual_val_loss(ckpt_path: Path) -> Optional[float]:
    """Load the actual val_loss from a checkpoint file.

    The filename contains a rounded val_loss (e.g., 0.0018), but the
    checkpoint file stores the precise value (e.g., 0.0017850291915237904).

    Args:
        ckpt_path: Path to the checkpoint file

    Returns:
        The actual val_loss or None if not found
    """
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Look for val_loss in ModelCheckpoint callback state
        if "callbacks" in ckpt:
            for cb_key, cb_val in ckpt["callbacks"].items():
                if "ModelCheckpoint" in cb_key and isinstance(cb_val, dict):
                    if "current_score" in cb_val:
                        return float(cb_val["current_score"])
                    if "best_model_score" in cb_val:
                        return float(cb_val["best_model_score"])

        return None
    except Exception as e:
        logger.warning(f"Could not load val_loss from {ckpt_path}: {e}")
        return None


def find_all_checkpoints(model: ModelConfig, variant: str) -> List[CheckpointInfo]:
    """Find all checkpoints for a model across all versions.

    Loads the actual val_loss from each checkpoint file to ensure
    accurate sorting (filename val_loss is rounded).

    Args:
        model: Model configuration
        variant: Either "dropout" or "no_dropout"

    Returns:
        List of CheckpointInfo objects, sorted by actual val_loss (best first)
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
            parsed = parse_checkpoint_name(ckpt_path)
            if parsed is None:
                continue

            epoch, filename_val_loss = parsed

            # Load actual val_loss from checkpoint file
            actual_val_loss = _load_actual_val_loss(ckpt_path)

            checkpoints.append(
                CheckpointInfo(
                    path=ckpt_path,
                    val_loss=filename_val_loss,
                    epoch=epoch,
                    version=version_num,
                    actual_val_loss=actual_val_loss,
                )
            )

    # Sort by best available val_loss (actual if available, otherwise filename)
    checkpoints.sort(key=lambda x: x.best_val_loss)
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
                           (legacy behavior, not recommended)

    Returns:
        Path to the best checkpoint, or None if not found
    """
    log_dir = get_log_dir(model, variant)

    if not log_dir.exists():
        return None

    if latest_version_only:
        # Legacy behavior: only search latest version
        versions = sorted(log_dir.glob("version_*"), key=_extract_version_number)
        if not versions:
            return None

        latest_version = versions[-1]
        checkpoints_dir = latest_version / "checkpoints"

        if not checkpoints_dir.exists():
            return None

        checkpoints = list(checkpoints_dir.glob("*.ckpt"))
        if not checkpoints:
            return None

        # Find checkpoint with lowest val_loss
        best_checkpoint = None
        best_loss = float("inf")

        for ckpt in checkpoints:
            parsed = parse_checkpoint_name(ckpt)
            if parsed is None:
                continue

            _, val_loss = parsed
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = ckpt

        return best_checkpoint

    # Default behavior: search all versions
    all_checkpoints = find_all_checkpoints(model, variant)
    if not all_checkpoints:
        return None

    return all_checkpoints[0].path
