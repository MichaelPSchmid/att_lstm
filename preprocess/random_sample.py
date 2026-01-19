import pickle
import random
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_preprocessed_paths

# Configuration
VEHICLE = "HYUNDAI_SONATA_2020"
paths = get_preprocessed_paths(VEHICLE, window_size=50, predict_size=1, step_size=1, suffix="sF_NewFeatures")

# File paths
feature_test_path = paths["features"]
target_test_path = paths["targets"]
sequence_ids_path = paths["sequence_ids"]
time_steps_path = paths["time_steps"]

# Load the saved data
with open(feature_test_path, 'rb') as x_file:
    X = pickle.load(x_file)

with open(target_test_path, 'rb') as y_file:
    Y = pickle.load(y_file)

with open(sequence_ids_path, 'rb') as id_file:
    sequence_ids = pickle.load(id_file)

with open(time_steps_path, 'rb') as ts_file:
    time_steps = pickle.load(ts_file)

# Check dataset size
print(f"Total Samples: {len(X)}")

# Randomly select a sample and print
if len(X) > 0:
    random_index = random.randint(0, len(X) - 1)
    print(f"\n🔹 Random Sample Index: {random_index}")
    print(f"🔹 Sequence ID: {sequence_ids[random_index]}")
    print(f"🔹 Starting Time Step: {time_steps[random_index]}")
    print(f"🔹 Sample Features (X):\n{X[random_index]}")
    print(f"🔹 Sample Target (Y):\n{Y[random_index]}")
