# Data Directory

This directory contains all data files for the EPS torque prediction project.

## Directory Structure

```
data/
├── dataset/                           # Raw CSV data
│   ├── HYUNDAI_SONATA_2020/          # Raw CSV files from vehicle
│   │   ├── *.csv                     # Individual driving segments
│   │   └── ...
│   └── TOYOTA_HIGHLANDER_2020/       # (Optional) Additional vehicle data
│       └── ...
│
└── prepared_dataset/                  # Preprocessed data (pickle files)
    ├── HYUNDAI_SONATA_2020/
    │   ├── {N}csv_with_sequence_id.pkl    # Concatenated raw data with sequence IDs
    │   ├── 50_1_1_sF/                     # Window=50, Predict=1, Step=1, suffix=sF
    │   │   ├── feature_50_1_1_sF.pkl      # Feature arrays (X)
    │   │   ├── target_50_1_1_sF.pkl       # Target arrays (Y)
    │   │   ├── sequence_ids_50_1_1_sF.pkl # Sequence IDs for traceability
    │   │   └── time_steps_50_1_1_sF.pkl   # Time stamps
    │   └── 15_1_1_s/                      # Alternative window size
    │       └── ...
    └── TOYOTA_HIGHLANDER_2020/
        └── ...
```

## Naming Convention

Preprocessed data folders follow this pattern:
```
{window_size}_{predict_size}_{step_size}_{suffix}
```

- `window_size`: Number of input time steps (e.g., 50)
- `predict_size`: Number of output time steps (e.g., 1)
- `step_size`: Sliding window step (e.g., 1)
- `suffix`: Variant identifier (e.g., "sF" for steerFiltered, "sF_NewFeatures" for extended features)

## Data Setup

1. Place raw CSV files in `data/dataset/HYUNDAI_SONATA_2020/`
2. Run preprocessing:
   ```bash
   python preprocess/data_preprocessing.py   # Creates {N}csv_with_sequence_id.pkl
   python preprocess/slice_window.py         # Creates feature/target pickle files
   ```

## Git Ignore

This directory is typically excluded from version control due to large file sizes.
Add to `.gitignore`:
```
data/dataset/
data/prepared_dataset/
*.pkl
```
