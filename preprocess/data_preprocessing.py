import pandas as pd
import os
from tqdm import tqdm

# Specify the folder path in Google Drive
folder_path = '/home/wudamu/MA_tianze/dataset/HYUNDAI_SONATA_2020'  # Change to your actual folder path

# Initialize an empty DataFrame
all_data = pd.DataFrame()

# Column names (ensure all CSV files use the same column names)
columns = [
    "vEgo", "aEgo", "steeringAngleDeg", "steeringPressed", "steer", "steerFiltered",
    "latActive", "roll", "t", "latAccelSteeringAngle", "latAccelDesired",
    "latAccelLocalizer", "epsFwVersion"
]

# Get and sort the list of files
files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')])

# Print the total number of CSV files
print(f"Total CSV files: {len(files)}")

# # Filter out duplicate file names (showing only the names)
# file_base_names = [os.path.splitext(f)[0].split(' ')[0] for f in files]  # Extract the base name, ignoring (1)
# duplicate_files = sorted(set([name for name in file_base_names if file_base_names.count(name) > 1]))

# # Print duplicate file names
# print("\nDuplicate files:")
# for duplicate in duplicate_files:
#     print(duplicate)

# Iterate over all CSV files and add sequence_id
for seq_id, file_name in tqdm(enumerate(files), total=len(files), desc='Processing files'):
    file_path = os.path.join(folder_path, file_name)

    try:
        # Read CSV into a DataFrame, specifying column names
        df = pd.read_csv(file_path, names=columns, header=0)  # Use header=0 to skip the first row of column names

        # Add sequence_id column
        df['sequence_id'] = seq_id

        # Concatenate to the main DataFrame
        all_data = pd.concat([all_data, df], ignore_index=True)
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

# Print the first 10 rows of the result
print(all_data.head(10))

# Save the merged data to Google Drive (optional)
output_path = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/20999csv_with_sequence_id.pkl'
all_data.to_pickle(output_path)
print(f"Saved merged data to {output_path}")
