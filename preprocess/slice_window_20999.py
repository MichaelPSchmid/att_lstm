import pickle
import pandas as pd
import sys
import os
from tqdm import tqdm

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_raw_data_path, PREPARED_DATASET_DIR

# Configuration
VEHICLE = "HYUNDAI_SONATA_2020"
NUM_CSVS = 20999

input_path = get_raw_data_path(VEHICLE, NUM_CSVS)
all_data = pd.read_pickle(input_path)

# Set window size and prediction size
window_size = 50
predict_size = 1
step_size = 1

# Select features and target
features = ['vEgo', 'aEgo', 'steeringAngleDeg', 'roll', 'latAccelLocalizer']
target = ['steerFiltered']

# Define paths
output_dir = PREPARED_DATASET_DIR / VEHICLE / f"{NUM_CSVS}csv" / f"{window_size}_{predict_size}_{step_size}_sF"
output_dir.mkdir(parents=True, exist_ok=True)

feature_test_path = output_dir / f"feature_{window_size}_{predict_size}_{step_size}_sF.pkl"
target_test_path = output_dir / f"target_{window_size}_{predict_size}_{step_size}_sF.pkl"
sequence_ids_path = output_dir / f"sequence_ids_{window_size}_{predict_size}_{step_size}_sF.pkl"
time_steps_path = output_dir / f"time_steps_{window_size}_{predict_size}_{step_size}_sF.pkl"


with open(feature_test_path, 'wb') as x_file, \
     open(target_test_path, 'wb') as y_file, \
     open(sequence_ids_path, 'wb') as id_file, \
     open(time_steps_path, 'wb') as ts_file:
    
    for i in tqdm(range(0, len(all_data) - window_size - predict_size + 1, step_size), desc="Processing samples"):
        window = all_data.iloc[i:i + window_size]
        predict = all_data.iloc[i + window_size:i + window_size + predict_size]

        window_sequence_ids = window['sequence_id'].unique()
        predict_sequence_ids = predict['sequence_id'].unique()

        if len(window_sequence_ids) == 1 and len(predict_sequence_ids) == 1 and \
           window_sequence_ids[0] == predict_sequence_ids[0]:  
            
            if ((window['latActive'] == True) & (window['steeringPressed'] == False)).all() and \
               ((predict['latActive'] == True) & (predict['steeringPressed'] == False)).all():
                
                
                pickle.dump(window[features].values, x_file)
                pickle.dump(predict[target].values, y_file)
                pickle.dump(window_sequence_ids[0], id_file)
                pickle.dump(window['t'].iloc[0], ts_file)
