# Evaluation Results - Dropout

> Generated: 2026-01-27 10:01:01

## Overview

| Model | Type | Parameters | FLOPs | R2 | Accuracy | RMSE | Inference (ms) |
|-------|------|------------|-------|-----|----------|------|----------------|
| M1 Small Baseline | LSTM (64, 3) | 84,801 | 4.314M | 0.844 | 80.49% | 0.0432 | 0.93 |
| M2 Small + Simple Attn | LSTM + Attention (64, 3) | 84,866 | 4.317M | 0.834 | 80.07% | 0.0445 | 1.01 |
| **M3 Medium Baseline** | LSTM (128, 5) | 597,633 | 30.131M | **0.893** | **86.29%** | 0.0357 | 2.62 |
| M4 Medium + Simple Attn | LSTM + Attention (128, 5) | 597,762 | 30.138M | 0.877 | 84.31% | 0.0382 | 3.45 |
| M5 Medium + Additive Attn | LSTM + Additive (128, 5) | 630,529 | 31.770M | 0.884 | 85.39% | 0.0372 | 4.73 |
| M6 Medium + Scaled DP | LSTM + Scaled DP (128, 5) | 597,633 | 30.131M | 0.885 | 85.23% | 0.0370 | 3.62 |

## Key Findings

- **Best Model:** M3 Medium Baseline (R2=0.893)
- **Variant:** With Dropout (0.2)
- **Models Evaluated:** 6/6

## Model Details

### M1 Small Baseline

- **Parameters:** 84,801
- **FLOPs:** 4.314M
- **MACs:** 2
- **R2:** 0.8436
- **Accuracy:** 80.49%
- **RMSE:** 0.0432
- **MAE:** 0.0318
- **Inference (P95):** 0.93 ms (single-thread)
- **Checkpoint:** `C:\Users\MSchm\Documents\att_project\lightning_logs\M1_Small_Baseline_Dropout\version_2\checkpoints\M1_Small_Baseline-epoch=43-val_loss=0.0019.ckpt`

### M2 Small + Simple Attn

- **Parameters:** 84,866
- **FLOPs:** 4.317M
- **MACs:** 2
- **R2:** 0.8341
- **Accuracy:** 80.07%
- **RMSE:** 0.0445
- **MAE:** 0.0323
- **Inference (P95):** 1.01 ms (single-thread)
- **Checkpoint:** `C:\Users\MSchm\Documents\att_project\lightning_logs\M2_Small_Simple_Attention_Dropout\version_0\checkpoints\M2_Small_Simple_Attention_Dropout-epoch=06-val_loss=0.0020.ckpt`

### M3 Medium Baseline

- **Parameters:** 597,633
- **FLOPs:** 30.131M
- **MACs:** 1
- **R2:** 0.8932
- **Accuracy:** 86.29%
- **RMSE:** 0.0357
- **MAE:** 0.0267
- **Inference (P95):** 2.62 ms (single-thread)
- **Checkpoint:** `C:\Users\MSchm\Documents\att_project\lightning_logs\M3_Medium_Baseline_Dropout\version_0\checkpoints\M3_Medium_Baseline_Dropout-epoch=59-val_loss=0.0013.ckpt`

### M4 Medium + Simple Attn

- **Parameters:** 597,762
- **FLOPs:** 30.138M
- **MACs:** 1
- **R2:** 0.8775
- **Accuracy:** 84.31%
- **RMSE:** 0.0382
- **MAE:** 0.0284
- **Inference (P95):** 3.45 ms (single-thread)
- **Checkpoint:** `C:\Users\MSchm\Documents\att_project\lightning_logs\M4_Medium_Simple_Attention_Dropout\version_0\checkpoints\M4_Medium_Simple_Attention_Dropout-epoch=27-val_loss=0.0015.ckpt`

### M5 Medium + Additive Attn

- **Parameters:** 630,529
- **FLOPs:** 31.770M
- **MACs:** 1
- **R2:** 0.8842
- **Accuracy:** 85.39%
- **RMSE:** 0.0372
- **MAE:** 0.0276
- **Inference (P95):** 4.73 ms (single-thread)
- **Checkpoint:** `C:\Users\MSchm\Documents\att_project\lightning_logs\M5_Medium_Additive_Attention_Dropout\version_0\checkpoints\M5_Medium_Additive_Attention_Dropout-epoch=34-val_loss=0.0014.ckpt`

### M6 Medium + Scaled DP

- **Parameters:** 597,633
- **FLOPs:** 30.131M
- **MACs:** 1
- **R2:** 0.8851
- **Accuracy:** 85.23%
- **RMSE:** 0.0370
- **MAE:** 0.0276
- **Inference (P95):** 3.62 ms (single-thread)
- **Checkpoint:** `C:\Users\MSchm\Documents\att_project\lightning_logs\M6_Medium_Scaled_DP_Attention_Dropout\version_0\checkpoints\M6_Medium_Scaled_DP_Attention_Dropout-epoch=30-val_loss=0.0014.ckpt`

---

*Results directory: `results/dropout/`*