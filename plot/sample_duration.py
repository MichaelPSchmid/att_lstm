import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_raw_data_path, PROJECT_ROOT

# Configure Matplotlib to use LaTeX and match LaTeX paper fonts
plt.rcParams.update({
    "text.usetex": True,        # Enable LaTeX text rendering
    "font.family": "serif",     # Use serif font
    "font.serif": ["Palatino"], # Make Matplotlib consistent with LaTeX
    "pdf.fonttype": 42,         # Ensure fonts in PDF are TrueType
    "ps.fonttype": 42,          # Same for PostScript
    "font.size": 15,            # Base font size
    "axes.labelsize": 15,       # Size for axis labels
    "legend.fontsize": 15,      # Size for legend text
    "xtick.labelsize": 15,      # Size for x-axis tick labels
    "ytick.labelsize": 15,      # Size for y-axis tick labels
})

def visualize_features(data, sequence_id):
    # Filter data based on sequence_id
    segment = data[data['sequence_id'] == sequence_id]
    
    # If no data found for the given sequence_id, display a message
    if segment.empty:
        print(f"No data found for sequence_id {sequence_id}.")
        return
    
    # Common plotting parameters
    plot_params = {
        'figsize': (10, 12),
        'alpha': 0.8,
        'grid': True
    }
    
    # Create the first figure with 3x1 subplots - first three features
    fig1, axs1 = plt.subplots(3, 1, figsize=plot_params['figsize'])
    
    # Define the plots for the first figure
    plots1 = [
        {'names': ['vEgo'], 'ax_idx': 0, 'title': r'Vehicle Speed (m/s)'},
        {'names': ['latAccelLocalizer'], 'ax_idx': 1, 'title': r'Lateral Acceleration (m/s$^2$)'},
        {'names': ['steeringAngleDeg'], 'ax_idx': 2, 'title': r'Steering Angle (degrees)'}
    ]
    
    # Plot data on the first figure
    for plot in plots1:
        for name in plot['names']:
            axs1[plot['ax_idx']].plot(segment['t'], segment[name], label=name, alpha=plot_params['alpha'])
        axs1[plot['ax_idx']].set_title(plot['title'])
        axs1[plot['ax_idx']].legend()
        axs1[plot['ax_idx']].grid(plot_params['grid'])
        axs1[plot['ax_idx']].set_xlabel(r'Time (s)')
    
    # Adjust layout and save the first figure
    plt.tight_layout()
    output_path1 = PROJECT_ROOT / "plot" / "output" / "primary_features.pdf"
    output_path1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path1, format='pdf', bbox_inches='tight')
    print(f"Saved plot to {output_path1}")
    plt.close(fig1)  # Close the first figure
    
    # Create the second figure with 3x1 subplots - remaining three features
    fig2, axs2 = plt.subplots(3, 1, figsize=plot_params['figsize'])
    
    # Define the plots for the second figure
    plots2 = [
        {'names': ['steer', 'steerFiltered'], 'ax_idx': 0, 'title': r'Steering Torque $[-1,1]$'},
        {'names': ['roll'], 'ax_idx': 1, 'title': r'Road Roll (rad)'},
        {'names': ['aEgo'], 'ax_idx': 2, 'title': r'Longitudinal Acceleration (m/s$^2$)'}
    ]
    
    # Plot data on the second figure
    for plot in plots2:
        for name in plot['names']:
            axs2[plot['ax_idx']].plot(segment['t'], segment[name], label=name, alpha=plot_params['alpha'])
        axs2[plot['ax_idx']].set_title(plot['title'])
        axs2[plot['ax_idx']].legend()
        axs2[plot['ax_idx']].grid(plot_params['grid'])
        axs2[plot['ax_idx']].set_xlabel(r'Time (s)')
    
    # Adjust layout and save the second figure
    plt.tight_layout()
    output_path2 = PROJECT_ROOT / "plot" / "output" / "secondary_features.pdf"
    plt.savefig(output_path2, format='pdf', bbox_inches='tight')
    print(f"Saved plot to {output_path2}")

    # Display the figures (if running in an interactive environment)
    plt.show()

# Load the data from the pickle file
data_file = get_raw_data_path("HYUNDAI_SONATA_2020", 5001)
data = pd.read_pickle(data_file)

# Visualize the data for sequence_id 4751
visualize_features(data, 4751)