import os
import pickle
from data_module import TimeSeriesDataModule

# File paths
feature_path = "/path/to/feature.pkl"
target_path = "/path/to/target.pkl"
sequence_ids_path = "/path/to/sequence_ids.pkl"
time_steps_path = "/path/to/time_steps.pkl"

# Load sequence_ids and time_steps
with open(sequence_ids_path, "rb") as f:
    sequence_ids = pickle.load(f)
with open(time_steps_path, "rb") as f:
    time_steps = pickle.load(f)

# Initialize the data module
data_module = TimeSeriesDataModule(feature_path, target_path, batch_size=32)

# Set up the data module
data_module.setup()

# Filter sequence_ids and time_steps for the test set
test_sequence_ids = [sequence_ids[i] for i in data_module.test_indices]
test_time_steps = [time_steps[i] for i in data_module.test_indices]

# Specify directory for saving test files
output_dir = "/path/to/save/test_data"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Save the filtered sequence_ids and time_steps
test_sequence_ids_path = os.path.join(output_dir, "test_sequence_ids.pkl")
test_time_steps_path = os.path.join(output_dir, "test_time_steps.pkl")

with open(test_sequence_ids_path, "wb") as f:
    pickle.dump(test_sequence_ids, f)
    print(f"Test sequence_ids saved to {test_sequence_ids_path}")

with open(test_time_steps_path, "wb") as f:
    pickle.dump(test_time_steps, f)
    print(f"Test time_steps saved to {test_time_steps_path}")
