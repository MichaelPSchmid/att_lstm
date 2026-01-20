"""
Central configuration for all paths and settings.

This module provides platform-independent paths using pathlib.
All scripts should import paths from here instead of hardcoding them.
"""

from pathlib import Path

# =============================================================================
# Base Paths
# =============================================================================

# Project root directory (where this file is located)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Data directory (inside project)
DATA_ROOT = PROJECT_ROOT / "data"

# Dataset paths
DATASET_DIR = DATA_ROOT / "dataset"
PREPARED_DATASET_DIR = DATA_ROOT / "prepared_dataset"  # Default (paper: 5001 files)
PREPARED_DATASET_FULL_DIR = DATA_ROOT / "prepared_dataset_full"  # Full dataset (all files)

# Output directories
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
LIGHTNING_LOGS_DIR = PROJECT_ROOT / "lightning_logs"
ATTENTION_VIS_DIR = PROJECT_ROOT / "attention_visualization"

# =============================================================================
# Vehicle-specific Paths
# =============================================================================

# Hyundai Sonata 2020
HYUNDAI_SONATA_RAW = DATASET_DIR / "HYUNDAI_SONATA_2020"
HYUNDAI_SONATA_PREPARED = PREPARED_DATASET_DIR / "HYUNDAI_SONATA_2020"

# Toyota Highlander 2020 (used in some plots)
TOYOTA_HIGHLANDER_RAW = DATASET_DIR / "TOYOTA_HIGHLANDER_2020"
TOYOTA_HIGHLANDER_PREPARED = PREPARED_DATASET_DIR / "TOYOTA_HIGHLANDER_2020"

# =============================================================================
# Preprocessed Data Paths
# =============================================================================

def get_preprocessed_paths(
    vehicle: str = "HYUNDAI_SONATA_2020",
    window_size: int = 50,
    predict_size: int = 1,
    step_size: int = 1,
    suffix: str = "sF",
    variant: str = "paper"
) -> dict:
    """
    Get paths for preprocessed data files.

    Args:
        vehicle: Vehicle name (folder name)
        window_size: Sliding window size
        predict_size: Prediction horizon
        step_size: Step size for sliding window
        suffix: Suffix for the folder (e.g., 'sF', 's', 'sF_NewFeatures')
        variant: Dataset variant - "paper" (5001 files) or "full" (all files)

    Returns:
        Dictionary with paths for features, targets, sequence_ids, time_steps
    """
    if variant == "full":
        base_dir = PREPARED_DATASET_FULL_DIR / vehicle
    else:
        base_dir = PREPARED_DATASET_DIR / vehicle
    folder_name = f"{window_size}_{predict_size}_{step_size}_{suffix}"
    data_dir = base_dir / folder_name

    file_base = f"{window_size}_{predict_size}_{step_size}_{suffix}"

    return {
        "dir": data_dir,
        "features": data_dir / f"feature_{file_base}.pkl",
        "targets": data_dir / f"target_{file_base}.pkl",
        "sequence_ids": data_dir / f"sequence_ids_{file_base}.pkl",
        "time_steps": data_dir / f"time_steps_{file_base}.pkl",
    }


def get_raw_data_path(vehicle: str, num_csvs: int) -> Path:
    """
    Get path to raw concatenated CSV data.

    Args:
        vehicle: Vehicle name
        num_csvs: Number of CSVs in the dataset (e.g., 5000, 20999)

    Returns:
        Path to the pickle file
    """
    return PREPARED_DATASET_DIR / vehicle / f"{num_csvs}csv_with_sequence_id.pkl"


# =============================================================================
# Default Paths (for backward compatibility)
# =============================================================================

# Default preprocessed data (most commonly used configuration)
# Using "full" variant for backward compatibility with existing data
DEFAULT_PATHS = get_preprocessed_paths(
    vehicle="HYUNDAI_SONATA_2020",
    window_size=50,
    predict_size=1,
    step_size=1,
    suffix="sF",
    variant="full"
)

# Convenience aliases
FEATURE_PATH = DEFAULT_PATHS["features"]
TARGET_PATH = DEFAULT_PATHS["targets"]

# =============================================================================
# Utility Functions
# =============================================================================

def ensure_dirs_exist():
    """Create all necessary directories if they don't exist."""
    dirs_to_create = [
        DATA_ROOT,
        DATASET_DIR,
        PREPARED_DATASET_DIR,
        EVALUATION_DIR,
        LIGHTNING_LOGS_DIR,
        ATTENTION_VIS_DIR,
        HYUNDAI_SONATA_RAW,
        HYUNDAI_SONATA_PREPARED,
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("Project Configuration")
    print("=" * 60)
    print(f"PROJECT_ROOT:         {PROJECT_ROOT}")
    print(f"DATA_ROOT:            {DATA_ROOT}")
    print(f"DATASET_DIR:          {DATASET_DIR}")
    print(f"PREPARED_DATASET_DIR: {PREPARED_DATASET_DIR}")
    print(f"EVALUATION_DIR:       {EVALUATION_DIR}")
    print(f"LIGHTNING_LOGS_DIR:   {LIGHTNING_LOGS_DIR}")
    print("-" * 60)
    print(f"FEATURE_PATH:         {FEATURE_PATH}")
    print(f"TARGET_PATH:          {TARGET_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
    print("\nCreating directories...")
    ensure_dirs_exist()
    print("Done!")
