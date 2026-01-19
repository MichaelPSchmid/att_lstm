import pickle
import pandas as pd
from tqdm import tqdm

input_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/20999csv_with_sequence_id.pkl'

all_data = pd.read_pickle(input_path)

# Set window size and prediction size
window_size = 50 
predict_size = 1
step_size = 1  

# Select features and target
features = ['vEgo', 'aEgo', 'steeringAngleDeg', 'roll', 'latAccelLocalizer']
target = ['steerFiltered']

# Define paths
feature_test_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/20999csv/50_1_1_sF/feature_50_1_1_sF.pkl'
target_test_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/20999csv/50_1_1_sF/target_50_1_1_sF.pkl'
sequence_ids_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/20999csv/50_1_1_sF/sequence_ids_50_1_1_sF.pkl'
time_steps_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/20999csv/50_1_1_sF/time_steps_50_1_1_sF.pkl'


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
