import pandas as pd
import matplotlib.pyplot as plt
import os

# Configure Matplotlib to use Arial font via LaTeX
plt.rcParams.update({
    "text.usetex": True,       # 启用LaTeX文本渲染
    "font.family": "sans-serif",  
    "text.latex.preamble": r"\usepackage{helvet} \renewcommand{\familydefault}{\sfdefault}",
    "pdf.fonttype": 42,        # 确保PDF中使用TrueType字体
    "ps.fonttype": 42,
    "font.size": 18,
    "axes.labelsize": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

# Read the CSV file
data_path = "/home/wudamu/MA_tianze/evaluation/15_1_1_s/test_results.csv"
df = pd.read_csv(data_path)

# Specify the target sequence_id for visualization
target_sequence_id = 4628  # Replace with the sequence_id you want to plot

# Filter the data based on sequence_id
filtered_df = df[df["sequence_id"] == target_sequence_id]

# Extract time_step, predictions, and targets
time_steps = filtered_df["time_step"].values
predictions = filtered_df["prediction"].apply(lambda x: float(x.strip("[]"))).values
targets = filtered_df["target"].apply(lambda x: float(x.strip("[]"))).values

# Plot the predictions and targets
plt.figure(figsize=(12, 6))
plt.plot(time_steps, predictions, label="Predictions", linestyle="-")
plt.plot(time_steps, targets, label="Targets", linestyle="-")

# Add titles and labels
plt.title(f"Predictions and Targets for A Random Sample")
plt.xlabel("Time Step (s)")
plt.ylabel("Normalized Steer Torque")
plt.legend()
plt.grid()

# Create save directory if it doesn't exist
save_dir = "/mnt/c/Users/wudamu/Desktop/MA/Thesis/chapter5/diff_attention/seed_3407/"
os.makedirs(save_dir, exist_ok=True)

# Save the figure with high resolution in the specified directory
plt.savefig(os.path.join(save_dir, f"sequence_{target_sequence_id}_comparison.pdf"), 
            bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(save_dir, f"sequence_{target_sequence_id}_comparison.png"), 
            bbox_inches='tight', dpi=300)

# Display the plot
plt.show()