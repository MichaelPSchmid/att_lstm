import pickle
import pandas as pd
import sys
import os
from tqdm import tqdm

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_raw_data_path, get_preprocessed_paths

# Configuration
VEHICLE = "HYUNDAI_SONATA_2020"
NUM_CSVS = 21000  # Number of CSV files in the dataset

input_path = get_raw_data_path(VEHICLE, NUM_CSVS)
all_data = pd.read_pickle(input_path)

# Set window size and prediction size
window_size = 50 
predict_size = 1
step_size = 1  # Sliding step size for each window 

# Select features and target
features = ['vEgo', 'aEgo', 'steeringAngleDeg', 'roll', 'latAccelLocalizer']
target = ['steerFiltered']

# Initialize X, Y, sequence_ids, and time_steps
X = []
Y = []
sequence_ids = []
time_steps = []  # To store the starting time step of each window

# Generate samples using a sliding window
total_steps = len(range(0, len(all_data) - window_size - predict_size + 1, step_size))
for i in tqdm(range(0, len(all_data) - window_size - predict_size + 1, step_size), desc="Processing samples", total=total_steps):
    # Extract the current window and prediction region (including sequence_id and all other columns)
    window = all_data.iloc[i:i + window_size]
    predict = all_data.iloc[i + window_size:i + window_size + predict_size]

    # Check if the data is within the same sequence_id
    window_sequence_ids = window['sequence_id'].unique()
    predict_sequence_ids = predict['sequence_id'].unique()

    if len(window_sequence_ids) == 1 and len(predict_sequence_ids) == 1 and \
       window_sequence_ids[0] == predict_sequence_ids[0]:  # Ensure both parts share the same sequence_id
        # Check if the window and prediction meet the conditions
        if ((window['latActive'] == True) & (window['steeringPressed'] == False)).all() and \
           ((predict['latActive'] == True) & (predict['steeringPressed'] == False)).all():
            # If conditions are met, save the input, output, sequence_id, and time step
            X.append(window[features].values)
            Y.append(predict[target].values)
            sequence_ids.append(window_sequence_ids[0])  # Save the corresponding sequence_id
            time_steps.append(window['t'].iloc[0])  # Save the starting time step of the window

# Print the number of samples
print(f"Number of samples: {len(X)}")

# Randomly select a sample and display its sequence_id and time_step
import random

if len(X) > 0:
    random_index = random.randint(0, len(X) - 1)
    print(f"Random sample index: {random_index}")
    print(f"Sequence ID: {sequence_ids[random_index]}")
    print(f"Starting Time Step: {time_steps[random_index]}")
    print(f"Sample features (X): {X[random_index]}")
    print(f"Sample target (Y): {Y[random_index]}")

# Save data
import pickle

# File paths for saving (from config)
paths = get_preprocessed_paths(VEHICLE, window_size, predict_size, step_size, "sF")
paths["dir"].mkdir(parents=True, exist_ok=True)

feature_test_path = paths["features"]
target_test_path = paths["targets"]
sequence_ids_path = paths["sequence_ids"]
time_steps_path = paths["time_steps"]

# Save X, Y, sequence_ids, and time_steps
with open(feature_test_path, 'wb') as x_file:
    pickle.dump(X, x_file)
    print(f"X data saved to {feature_test_path}")

with open(target_test_path, 'wb') as y_file:
    pickle.dump(Y, y_file)
    print(f"Y data saved to {target_test_path}")

with open(sequence_ids_path, 'wb') as id_file:
    pickle.dump(sequence_ids, id_file)
    print(f"Sequence IDs saved to {sequence_ids_path}")

with open(time_steps_path, 'wb') as ts_file:
    pickle.dump(time_steps, ts_file)
    print(f"Time Steps saved to {time_steps_path}")