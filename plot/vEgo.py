import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 配置Matplotlib使用LaTeX和Arial字体
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
    
    # 保存为JPG格式
    plt.savefig('/mnt/c/Users/wudamu/Desktop/MA/Thesis/chapter3/plot/vEgo.jpg', 
                format='jpg', bbox_inches='tight', dpi=300)
    
    # 显示图形（如果在交互式环境中运行）
    plt.show()

# 从pickle文件加载数据
data_file = '/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/5001csv_with_sequence_id.pkl'
data = pd.read_pickle(data_file)

# 可视化sequence_id为4751的数据
visualize_vego(data, 4751)