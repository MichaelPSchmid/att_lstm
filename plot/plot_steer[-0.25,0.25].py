import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
input_path = '/home/wudamu/MA_tianze/prepared_dataset/TOYOTA_HIGHLANDER_2020/20399csv_with_sequence_id.pkl'
all_data = pd.read_pickle(input_path)

# Define interval bins (refined range from -0.25 to 0.25 with 0.05 steps)
bins = [-0.25 + i * 0.05 for i in range(11)]  # Generate intervals: [-0.25, -0.20, ..., 0.20, 0.25]

# Calculate frequency for each bin
hist, bin_edges = np.histogram(all_data['steer'], bins=bins)

# Calculate probabilities for each bin
probabilities = hist / len(all_data)

# Print intervals and probabilities
print("Steer Value Probability Distribution (-0.25 to 0.25, Step 0.05):")
for i in range(len(bins) - 1):
    print(f"Interval [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {probabilities[i]:.4f}")

# Plot the probability distribution as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], probabilities, width=0.05, align='edge', color='green', edgecolor='black', alpha=0.7)

# Add labels and title
plt.xlabel('Steer Value Ranges (-0.25 to 0.25)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Probability Distribution of Steer Values (Step 0.05)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(bin_edges, rotation=45)

# Show the chart
plt.tight_layout()
plt.show()
