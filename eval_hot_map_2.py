import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model.attention_visualization.scaled_dot_product_attention_visualize import LSTMScaledDotAttentionModel
from data_module import TimeSeriesDataModule
import os

# 设置随机种子确保实验可重复性
pl.seed_everything(3407)
torch.set_float32_matmul_precision('medium')

# 文件路径
checkpoint_dir = "/home/wudamu/MA_tianze/lightning_logs/LSTMScaledDotAttentionModel/win50_128_5_seed3407/checkpoints"
best_checkpoint = os.path.join(checkpoint_dir, "LSTMScaledDotAttentionModel-epoch=60-val_loss=0.0011.ckpt")
feature_path = "/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF/feature_50_1_1_sF.pkl"
target_path = "/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF/target_50_1_1_sF.pkl"
output_dir = "attention_visualization/matrix"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
data_module = TimeSeriesDataModule(feature_path, target_path, batch_size=32)
data_module.setup("test")
test_dataloader = data_module.test_dataloader()

# 加载模型
model = LSTMScaledDotAttentionModel.load_from_checkpoint(best_checkpoint, strict=False)
model.eval()

# # ---------- ✅ 可视化函数：单样本 ----------
# def visualize_attention_matrix(model, x, sample_idx=0, seq_length=None, save_path=None):
#     with torch.no_grad():
#         _, attention_weights = model(x, return_attention=True)
#     attention_matrix = attention_weights[sample_idx].cpu().numpy()
#     if seq_length is None:
#         seq_length = attention_matrix.shape[0]
#     time_labels = [f"t-{seq_length-i}" for i in range(seq_length)]
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(attention_matrix, annot=True, fmt=".2f", cmap="Blues",
#                 xticklabels=time_labels, yticklabels=time_labels)
#     plt.title('Attention Matrix')
#     plt.xlabel('Key (attended to)')
#     plt.ylabel('Query (attending from)')
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#         print(f"Saved attention matrix visualization to {save_path}")
#     plt.show()
#     return attention_matrix

# ---------- ✅ 新增函数：平均注意力矩阵 ----------
def visualize_average_attention_matrix(model, dataloader, save_path=None):
    all_attentions = []
    with torch.no_grad():
        for batch in dataloader:
            X_batch, _ = batch
            _, attention_weights = model(X_batch, return_attention=True)
            all_attentions.append(attention_weights.cpu())

    all_attentions_tensor = torch.cat(all_attentions, dim=0)  # (N, S, S)
    avg_attention = torch.mean(all_attentions_tensor, dim=0)  # (S, S)

    # 保存为 numpy 文件
    avg_attention_np = avg_attention.numpy()
    np.save("attention_visualization/matrix/avg_attention.npy", avg_attention_np)
    print("✅ Saved avg_attention_np to attention_visualization/matrix/avg_attention.npy")

    seq_len = avg_attention_np.shape[0]
    time_labels = [f"t-{seq_len - i}" for i in range(seq_len)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attention_np, annot=False, cmap="YlGnBu",
                xticklabels=time_labels, yticklabels=time_labels)
    plt.title("Average Attention Matrix (Test Set)")
    plt.xlabel("Key (attended to)")
    plt.ylabel("Query (attending from)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved average attention heatmap to {save_path}")
    plt.show()

# # ---------- ✅ 可视化：单个样本 ----------
# first_batch = next(iter(test_dataloader))
# X_test, _ = first_batch
# visualize_attention_matrix(
#     model, 
#     X_test, 
#     sample_idx=0, 
#     save_path=os.path.join(output_dir, "attention_matrix_sample0.png")
# )

# ---------- ✅ 可视化：平均注意力矩阵 ----------
visualize_average_attention_matrix(
    model, 
    test_dataloader, 
    save_path=os.path.join(output_dir, "average_attention_matrix.png")
)
