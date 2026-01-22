import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_raw_data_path, PROJECT_ROOT

# Configure Matplotlib to use LaTeX and Arial font
plt.rcParams.update({
    "text.usetex": True,        # 启用LaTeX文本渲染
    "font.family": "sans-serif", # 指定使用无衬线字体族
    "text.latex.preamble": r"\usepackage{helvet} \renewcommand{\familydefault}{\sfdefault}", # 使用helvet包提供的Helvetica (类似Arial)
    "font.size": 15,           # 基本字体大小
    "axes.labelsize": 15,      # 坐标轴标签大小
    "legend.fontsize": 15,     # 图例文本大小
    "xtick.labelsize": 15,     # x轴刻度标签大小
    "ytick.labelsize": 15,     # y轴刻度标签大小
})

def visualize_vego(data, sequence_id):
    # 根据sequence_id筛选数据
    segment = data[data['sequence_id'] == sequence_id]
    
    # 如果没有找到对应sequence_id的数据，显示消息
    if segment.empty:
        print(f"No data found for sequence_id {sequence_id}.")
        return
    
    # 创建图形
    plt.figure(figsize=(10, 4))
    
    # 绘制vEgo数据
    plt.plot(segment['t'], segment['vEgo'], label='vEgo', alpha=0.8)
    plt.title('Vehicle Speed (m/s)')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    
    # 调整布局
    plt.tight_layout()
    
    # Save as JPG format
    output_path = PROJECT_ROOT / "plot" / "output" / "vEgo.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='jpg', bbox_inches='tight', dpi=300)
    print(f"Saved plot to {output_path}")

    # Show figure (if running in interactive environment)
    plt.show()

# Load data from pickle file
data_file = get_raw_data_path("HYUNDAI_SONATA_2020", 5001)
data = pd.read_pickle(data_file)

# Visualize data for sequence_id 4751
visualize_vego(data, 4751)