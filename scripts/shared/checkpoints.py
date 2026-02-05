"""
Checkpoint discovery and management utilities.

Provides consistent checkpoint finding logic across all evaluation scripts.
The key design decision: search ALL versions for the best checkpoint,
not just the latest version.

Supports multi-seed training: finds checkpoints for base model and all
seed variants (e.g., M4_Small_Simple_Attention, M4_Small_Simple_Attention_seed123).
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import ModelConfig
from .paths import get_log_dir, PROJECT_ROOT

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


def find_seed_variants(model: ModelConfig, variant: str) -> Dict[int, Path]:
    """Find all seed variants for a model in lightning_logs.

    Searches for directories matching the pattern {base_name}_seed{N}.
    All seed variants must have explicit _seedN suffix.

    Args:
        model: Model configuration
        variant: Either "dropout" or "no_dropout"

    Returns:
        Dictionary mapping seed -> log_dir path.
    """
    base_log_dir = get_log_dir(model, variant)
    base_name = base_log_dir.name
    lightning_logs = PROJECT_ROOT / "lightning_logs"

    if not lightning_logs.exists():
        return {}

    seed_variants: Dict[int, Path] = {}

    # Pattern to match seed suffix (required)
    seed_pattern = re.compile(rf"^{re.escape(base_name)}_seed(\d+)$")

    for log_dir in lightning_logs.iterdir():
        if not log_dir.is_dir():
            continue

        match = seed_pattern.match(log_dir.name)
        if match:
            seed = int(match.group(1))
            seed_variants[seed] = log_dir

    return seed_variants


def find_best_checkpoint_for_seed(
    model: ModelConfig,
    variant: str,
    seed: int,
) -> Optional[Path]:
    """Find the best checkpoint for a specific seed variant.

    Args:
        model: Model configuration
        variant: Either "dropout" or "no_dropout"
        seed: The seed value (42 = base model)

    Returns:
        Path to the best checkpoint, or None if not found
    """
    seed_variants = find_seed_variants(model, variant)

    if seed not in seed_variants:
        return None

    log_dir = seed_variants[seed]
    checkpoints = []

    # Search all version directories in this seed variant
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
                continue

            epoch, val_loss = metadata
            checkpoints.append((val_loss, ckpt_path))

    if not checkpoints:
        return None

    # Return checkpoint with lowest val_loss
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[0][1]


def find_all_seed_checkpoints(
    model: ModelConfig,
    variant: str,
) -> Dict[int, Tuple[Path, float]]:
    """Find the best checkpoint for each seed variant.

    Args:
        model: Model configuration
        variant: Either "dropout" or "no_dropout"

    Returns:
        Dictionary mapping seed -> (checkpoint_path, val_loss)
    """
    seed_variants = find_seed_variants(model, variant)
    result: Dict[int, Tuple[Path, float]] = {}

    for seed, log_dir in seed_variants.items():
        checkpoints = []

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
                    continue

                epoch, val_loss = metadata
                checkpoints.append((val_loss, ckpt_path))

        if checkpoints:
            checkpoints.sort(key=lambda x: x[0])
            best_val_loss, best_path = checkpoints[0]
            result[seed] = (best_path, best_val_loss)

    return result


def find_latest_checkpoint_for_resume(
    model: ModelConfig,
    variant: str,
    seed: Optional[int] = None,
) -> Optional[Tuple[Path, int, float]]:
    """Find the latest checkpoint for resuming interrupted training.

    Takes the highest version directory, then the checkpoint with the highest
    epoch number. Parses epoch/val_loss from the filename to avoid slow
    torch.load calls (the rounded val_loss is sufficient for display purposes).

    Args:
        model: Model configuration
        variant: Either "dropout" or "no_dropout"
        seed: Optional seed value. If provided, looks for {model_name}_seed{N} directory.
              If None, looks for the base model directory.

    Returns:
        Tuple of (checkpoint_path, epoch, val_loss) or None if not found
    """
    # Determine which log directory to search
    if seed is not None:
        seed_variants = find_seed_variants(model, variant)
        if seed not in seed_variants:
            return None
        log_dir = seed_variants[seed]
    else:
        log_dir = get_log_dir(model, variant)

    if not log_dir.exists():
        return None

    # Find the highest version directory
    version_dirs = []
    for version_dir in log_dir.glob("version_*"):
        version_num = _extract_version_number(version_dir)
        if version_num >= 0:
            version_dirs.append((version_num, version_dir))

    if not version_dirs:
        return None

    version_dirs.sort(reverse=True)
    latest_version_dir = version_dirs[0][1]

    # Find the checkpoint with the highest epoch in that version
    checkpoints_dir = latest_version_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None

    epoch_pattern = re.compile(r"epoch=(\d+)-val_loss=([\d.]+)")
    checkpoints = []

    for ckpt_path in checkpoints_dir.glob("*.ckpt"):
        match = epoch_pattern.search(ckpt_path.stem)
        if match:
            epoch = int(match.group(1))
            val_loss = float(match.group(2))
            checkpoints.append((epoch, val_loss, ckpt_path))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_epoch, latest_val_loss, latest_path = checkpoints[0]

    return (latest_path, latest_epoch, latest_val_loss)
