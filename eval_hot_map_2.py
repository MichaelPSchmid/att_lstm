import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model.attention_visualization.scaled_dot_product_attention_visualize import LSTMScaledDotAttentionModel
from data_module import TimeSeriesDataModule

from config import FEATURE_PATH, TARGET_PATH, LIGHTNING_LOGS_DIR, ATTENTION_VIS_DIR

# Set random seed for reproducibility
pl.seed_everything(3407)
torch.set_float32_matmul_precision('medium')

# File paths (from config)
checkpoint_dir = LIGHTNING_LOGS_DIR / "LSTMScaledDotAttentionModel" / "win50_128_5_seed3407" / "checkpoints"
best_checkpoint = checkpoint_dir / "LSTMScaledDotAttentionModel-epoch=60-val_loss=0.0011.ckpt"
feature_path = FEATURE_PATH
target_path = TARGET_PATH
output_dir = ATTENTION_VIS_DIR / "matrix"
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
data_module = TimeSeriesDataModule(str(feature_path), str(target_path), batch_size=32)
data_module.setup("test")
test_dataloader = data_module.test_dataloader()

# Load model
model = LSTMScaledDotAttentionModel.load_from_checkpoint(str(best_checkpoint), strict=False)
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

    # Save as numpy file
    avg_attention_np = avg_attention.numpy()
    save_npy_path = output_dir / "avg_attention.npy"
    np.save(save_npy_path, avg_attention_np)
    print(f"Saved avg_attention_np to {save_npy_path}")

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

# ---------- Visualize average attention matrix ----------
visualize_average_attention_matrix(
    model,
    test_dataloader,
    save_path=str(output_dir / "average_attention_matrix.png")
)
