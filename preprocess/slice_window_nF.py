import pickle
import pandas as pd
from tqdm import tqdm
import pickle
import os

# Load the dataset
input_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/5001csv_with_sequence_id.pkl'
all_data = pd.read_pickle(input_path)

# Compute the first and second derivatives of `steeringAngleDeg`
all_data['AngVel'] = all_data['steeringAngleDeg'].diff()  # First derivative
all_data['AngAcc'] = all_data['AngVel'].diff()  # Second derivative

# Fill NaN values (caused by `diff()`) with zero for safety
all_data.fillna(0, inplace=True)

# Set window size and prediction size
window_size = 50 
predict_size = 1
step_size = 1  # Sliding step size for each window 

# Select features and target
features = ['vEgo', 'aEgo', 'steeringAngleDeg', 'roll', 'latAccelLocalizer', 
            'AngVel', 'AngAcc']  # Added derivatives
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
            X.append(window[features].values)  # Include new features
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


# File paths for saving
feature_test_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF_NewFeatures/feature_50_1_1_sF_nF.pkl'
target_test_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF_NewFeatures/target_50_1_1_sF_nF.pkl'
sequence_ids_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF_NewFeatures/sequence_ids_50_1_1_sF_nF.pkl'
time_steps_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF_NewFeatures/time_steps_50_1_1_sF_nF.pkl'

os.makedirs(os.path.dirname(feature_test_path), exist_ok=True)

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
