import pandas as pd
import pickle

test_predictions_path = "/home/wudamu/MA_tianze/evaluation/test_predictions.pkl"
test_targets_path = "/home/wudamu/MA_tianze/evaluation/test_targets.pkl"

with open(test_predictions_path, "rb") as f:
    test_predictions = pickle.load(f)
with open(test_targets_path, "rb") as f:
    test_targets = pickle.load(f)

sequence_ids_path = "/home/wudamu/MA_tianze/evaluation/test_sequence_ids.pkl"
time_steps_path = "/home/wudamu/MA_tianze/evaluation/test_time_steps.pkl"

with open(sequence_ids_path, "rb") as f:
    sequence_ids = pickle.load(f)
with open(time_steps_path, "rb") as f:
    time_steps = pickle.load(f)

# update and truncate time_steps
updated_time_steps = [round(t + 1.5, 2) for t in time_steps]
time_steps = updated_time_steps

assert len(sequence_ids) == len(time_steps) == len(test_predictions) == len(test_targets), "Data lengths do not match!"


data = {
    "sequence_id": sequence_ids,
    "time_step": time_steps,
    "prediction": test_predictions,  
    "target": test_targets           
}

df = pd.DataFrame(data)


print(df.head())

csv_path = "/home/wudamu/MA_tianze/evaluation/test_results.csv"
pkl_path = "/home/wudamu/MA_tianze/evaluation/test_results.pkl"

import os
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

df.to_csv(csv_path, index=False)
print(f"DataFrame saved to '{csv_path}'.")

df.to_pickle(pkl_path)
print(f"DataFrame saved to '{pkl_path}'.")


