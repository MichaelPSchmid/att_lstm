import pickle
import random

# File paths
feature_test_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF_NewFeatures/feature_50_1_1_sF_nF.pkl'
target_test_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF_NewFeatures/target_50_1_1_sF_nF.pkl'
sequence_ids_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF_NewFeatures/sequence_ids_50_1_1_sF_nF.pkl'
time_steps_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF_NewFeatures/time_steps_50_1_1_sF_nF.pkl'

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
