"""
Parallel preprocessing script for sliding window extraction.

This script combines data_preprocessing.py and slice_window.py into a single,
more efficient pipeline that processes CSV files in parallel without needing
to merge them first.

Uses memory-efficient numpy .npy format instead of pickle for large arrays.
Supports chunked processing to handle datasets larger than available RAM.

Usage:
    python preprocess/preprocess_parallel.py

Configuration can be adjusted via command-line arguments or by modifying
the constants below.
"""

import argparse
import gc
import logging
import pickle
import sys
import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    HYUNDAI_SONATA_RAW,
    get_preprocessed_paths,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PreprocessConfig:
    """Configuration for preprocessing."""

    # Input/Output
    vehicle: str = "HYUNDAI_SONATA_2020"
    raw_data_path: Path = HYUNDAI_SONATA_RAW

    # Sliding window parameters
    window_size: int = 50
    predict_size: int = 1
    step_size: int = 1

    # Feature configuration
    features: Tuple[str, ...] = (
        "vEgo", "aEgo", "steeringAngleDeg", "roll", "latAccelLocalizer"
    )
    target: str = "steerFiltered"
    suffix: str = "sF"

    # Filter conditions
    require_lat_active: bool = True
    require_no_steering_pressed: bool = True

    # Processing
    num_workers: Optional[int] = None  # None = auto (cpu_count - 4)
    max_files: Optional[int] = None  # None = use all files

    # CSV column names
    csv_columns: Tuple[str, ...] = (
        "vEgo", "aEgo", "steeringAngleDeg", "steeringPressed", "steer",
        "steerFiltered", "latActive", "roll", "t", "latAccelSteeringAngle",
        "latAccelDesired", "latAccelLocalizer", "epsFwVersion"
    )


# =============================================================================
# Processing Functions
# =============================================================================

def process_single_csv(
    args: Tuple[int, Path],
    config: PreprocessConfig
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[float]]:
    """
    Process a single CSV file and extract sliding windows.

    Args:
        args: Tuple of (sequence_id, csv_path)
        config: Preprocessing configuration

    Returns:
        Tuple of (X_samples, Y_samples, sequence_ids, time_steps)
    """
    seq_id, csv_path = args

    X_samples = []
    Y_samples = []
    sequence_ids = []
    time_steps = []

    try:
        # Read CSV
        df = pd.read_csv(csv_path, names=list(config.csv_columns), header=0)

        # Apply filters
        if config.require_lat_active:
            mask = df["latActive"] == True
        else:
            mask = pd.Series([True] * len(df))

        if config.require_no_steering_pressed:
            mask = mask & (df["steeringPressed"] == False)

        # Find contiguous valid segments
        df["valid"] = mask
        df["segment"] = (df["valid"] != df["valid"].shift()).cumsum()

        # Process each valid segment
        valid_segments = df[df["valid"]].groupby("segment")

        for _, segment_df in valid_segments:
            segment_len = len(segment_df)
            min_len = config.window_size + config.predict_size

            if segment_len < min_len:
                continue

            # Extract windows from this segment
            feature_data = segment_df[list(config.features)].values
            target_data = segment_df[[config.target]].values
            time_data = segment_df["t"].values

            for i in range(0, segment_len - min_len + 1, config.step_size):
                X_samples.append(feature_data[i:i + config.window_size])
                Y_samples.append(target_data[i + config.window_size:i + min_len])
                sequence_ids.append(seq_id)
                time_steps.append(time_data[i])

    except Exception as e:
        logger.warning(f"Error processing {csv_path.name}: {e}")

    return X_samples, Y_samples, sequence_ids, time_steps


def run_parallel_preprocessing(config: PreprocessConfig) -> None:
    """
    Run the parallel preprocessing pipeline.

    Args:
        config: Preprocessing configuration
    """
    # Get list of CSV files
    csv_files = sorted([
        f for f in config.raw_data_path.iterdir()
        if f.is_file() and f.suffix == ".csv"
    ])

    if not csv_files:
        logger.error(f"No CSV files found in {config.raw_data_path}")
        return

    logger.info(f"Found {len(csv_files)} CSV files")

    # Limit number of files if specified
    if config.max_files is not None and config.max_files < len(csv_files):
        csv_files = csv_files[:config.max_files]
        logger.info(f"Limited to first {config.max_files} CSV files")

    # Prepare arguments: (sequence_id, path)
    work_items = list(enumerate(csv_files))

    # Determine number of workers
    num_workers = config.num_workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 4)
    logger.info(f"Using {num_workers} worker processes")

    # Process in parallel
    process_func = partial(process_single_csv, config=config)

    all_X = []
    all_Y = []
    all_seq_ids = []
    all_time_steps = []

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, work_items),
            total=len(work_items),
            desc="Processing CSVs"
        ))

    # Determine dataset variant based on whether max_files was used
    variant = "paper" if config.max_files is not None else "full"

    # Get output paths
    paths = get_preprocessed_paths(
        vehicle=config.vehicle,
        window_size=config.window_size,
        predict_size=config.predict_size,
        step_size=config.step_size,
        suffix=config.suffix,
        variant=variant
    )

    # Create output directory
    paths["dir"].mkdir(parents=True, exist_ok=True)

    # Count total samples first
    total_samples = sum(len(X) for X, _, _, _ in results)
    logger.info(f"Total samples extracted: {total_samples}")

    if total_samples == 0:
        logger.error("No samples extracted! Check your filter conditions.")
        return

    # Save using memory-efficient chunked approach
    logger.info(f"Saving to {paths['dir']} (using numpy format)")

    # Define .npy paths
    features_npy = paths["dir"] / f"features_{config.window_size}_{config.predict_size}_{config.step_size}_{config.suffix}.npy"
    targets_npy = paths["dir"] / f"targets_{config.window_size}_{config.predict_size}_{config.step_size}_{config.suffix}.npy"

    # Pre-allocate memory-mapped arrays for efficient writing
    feature_shape = (total_samples, config.window_size, len(config.features))
    target_shape = (total_samples, config.predict_size)

    logger.info(f"Creating memory-mapped arrays: features {feature_shape}, targets {target_shape}")

    # Use memory-mapped files for writing (avoids loading everything into RAM)
    features_mmap = np.lib.format.open_memmap(
        features_npy, mode='w+', dtype=np.float64, shape=feature_shape
    )
    targets_mmap = np.lib.format.open_memmap(
        targets_npy, mode='w+', dtype=np.float64, shape=target_shape
    )

    # Also collect sequence_ids and time_steps (these are small)
    all_seq_ids = []
    all_time_steps = []

    # Write data chunk by chunk
    logger.info("Writing data to disk...")
    idx = 0
    for X, Y, seq_ids, times in tqdm(results, desc="Saving chunks"):
        if len(X) == 0:
            continue

        chunk_size = len(X)
        features_mmap[idx:idx + chunk_size] = np.array(X)
        targets_mmap[idx:idx + chunk_size] = np.array(Y).reshape(-1, config.predict_size)
        all_seq_ids.extend(seq_ids)
        all_time_steps.extend(times)
        idx += chunk_size

    # Flush memory-mapped files
    del features_mmap
    del targets_mmap
    gc.collect()

    logger.info(f"Features saved: {features_npy}")
    logger.info(f"Targets saved: {targets_npy}")

    # Save sequence_ids and time_steps as pickle (small data)
    with open(paths["sequence_ids"], "wb") as f:
        pickle.dump(all_seq_ids, f)
    logger.info(f"Sequence IDs saved: {paths['sequence_ids']}")

    with open(paths["time_steps"], "wb") as f:
        pickle.dump(all_time_steps, f)
    logger.info(f"Time steps saved: {paths['time_steps']}")

    # Note: Pickle files are NOT created by default for large datasets
    # as they require loading everything into RAM. Use numpy files instead.
    # The data_module.py has been updated to support numpy format.

    # Print summary
    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"  Dataset variant: {variant}")
    logger.info(f"  Samples: {total_samples}")
    logger.info(f"  Feature shape: {feature_shape}")
    logger.info(f"  Target shape: {target_shape}")
    logger.info(f"  Unique sequences: {len(set(all_seq_ids))}")
    logger.info(f"  Output files:")
    logger.info(f"    - {features_npy.name}")
    logger.info(f"    - {targets_npy.name}")
    logger.info(f"    - {paths['sequence_ids'].name}")
    logger.info(f"    - {paths['time_steps'].name}")
    logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Parallel preprocessing for sliding window extraction"
    )

    parser.add_argument(
        "--window-size", type=int, default=50,
        help="Sliding window size (default: 50)"
    )
    parser.add_argument(
        "--predict-size", type=int, default=1,
        help="Prediction horizon (default: 1)"
    )
    parser.add_argument(
        "--step-size", type=int, default=1,
        help="Step size for sliding window (default: 1)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of worker processes (default: cpu_count - 4)"
    )
    parser.add_argument(
        "--vehicle", type=str, default="HYUNDAI_SONATA_2020",
        help="Vehicle name (default: HYUNDAI_SONATA_2020)"
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Maximum number of CSV files to process (default: all)"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    config = PreprocessConfig(
        window_size=args.window_size,
        predict_size=args.predict_size,
        step_size=args.step_size,
        num_workers=args.workers,
        vehicle=args.vehicle,
        max_files=args.max_files,
    )

    # Set raw data path based on vehicle
    if args.vehicle == "HYUNDAI_SONATA_2020":
        config.raw_data_path = HYUNDAI_SONATA_RAW
    else:
        from config.settings import DATASET_DIR
        config.raw_data_path = DATASET_DIR / args.vehicle

    run_parallel_preprocessing(config)


if __name__ == "__main__":
    main()
