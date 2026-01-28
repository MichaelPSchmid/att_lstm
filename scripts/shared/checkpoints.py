"""
Checkpoint discovery and management utilities.

Provides consistent checkpoint finding logic across all evaluation scripts.
The key design decision: search ALL versions for the best checkpoint,
not just the latest version.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .models import ModelConfig
from .paths import get_log_dir


@dataclass
class CheckpointInfo:
    """Information about a discovered checkpoint."""

    path: Path
    val_loss: float
    epoch: int
    version: int

    @property
    def name(self) -> str:
        """Return the checkpoint filename."""
        return self.path.name


def parse_checkpoint_name(path: Path) -> Optional[tuple]:
    """Parse checkpoint filename to extract metadata.

    Expected format: ModelName-epoch=XX-val_loss=X.XXXX.ckpt

    Args:
        path: Path to the checkpoint file

    Returns:
        Tuple of (epoch, val_loss) or None if parsing fails
    """
    name = path.stem
    try:
        # Extract epoch
        epoch_part = name.split("epoch=")[1].split("-")[0]
        epoch = int(epoch_part)

        # Extract val_loss
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


def find_all_checkpoints(model: ModelConfig, variant: str) -> List[CheckpointInfo]:
    """Find all checkpoints for a model across all versions.

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
            parsed = parse_checkpoint_name(ckpt_path)
            if parsed is None:
                continue

            epoch, val_loss = parsed
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
