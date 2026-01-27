# Evaluation Results - No Dropout

> Generated: 2026-01-27 09:22:40

## Overview

| Model | Type | Parameters | FLOPs | R2 | Accuracy | RMSE | Inference (ms) |
|-------|------|------------|-------|-----|----------|------|----------------|
| M1 Small Baseline | LSTM (64, 3) | 84,801 | 4.314M | 0.860 | 82.57% | 0.0408 | 0.92 |
| M2 Small + Simple Attn | LSTM + Attention (64, 3) | 84,866 | 4.317M | 0.850 | 81.50% | 0.0423 | 0.97 |
| M3 Medium Baseline | LSTM (128, 5) | 597,633 | 30.131M | 0.905 | 87.84% | 0.0338 | 2.66 |
| **M4 Medium + Simple Attn** | LSTM + Attention (128, 5) | 597,762 | 30.138M | **0.919** | **90.25%** | 0.0311 | 2.75 |
| M5 Medium + Additive Attn | LSTM + Additive (128, 5) | 630,529 | 31.770M | 0.907 | 88.34% | 0.0332 | 3.82 |
| M6 Medium + Scaled DP | LSTM + Scaled DP (128, 5) | 597,633 | 30.131M | 0.907 | 88.17% | 0.0334 | 2.77 |

## Key Findings

- **Best Model:** M4 Medium + Simple Attn (R2=0.919)
- **Variant:** No Dropout
- **Models Evaluated:** 6/6

## Model Details

### M1 Small Baseline

- **Parameters:** 84,801
- **FLOPs:** 4.314M
- **MACs:** 2
- **R2:** 0.8604
- **Accuracy:** 82.57%
- **RMSE:** 0.0408
- **MAE:** 0.0300
- **Inference (P95):** 0.92 ms (single-thread)
- **Checkpoint:** `C:\Users\MSchm\Documents\att_project\lightning_logs\M1_Small_Baseline\version_0\checkpoints\M1_Small_Baseline-epoch=12-val_loss=0.0017.ckpt`

### M2 Small + Simple Attn

- **Parameters:** 84,866
- **FLOPs:** 4.317M
- **MACs:** 2
- **R2:** 0.8503
- **Accuracy:** 81.50%
- **RMSE:** 0.0423
- **MAE:** 0.0309
- **Inference (P95):** 0.97 ms (single-thread)
- **Checkpoint:** `C:\Users\MSchm\Documents\att_project\lightning_logs\M2_Small_Simple_Attention\version_0\checkpoints\M2_Small_Simple_Attention-epoch=06-val_loss=0.0018.ckpt`

### M3 Medium Baseline

- **Parameters:** 597,633
- **FLOPs:** 30.131M
- **MACs:** 1
- **R2:** 0.9046
- **Accuracy:** 87.84%
- **RMSE:** 0.0338
- **MAE:** 0.0255
- **Inference (P95):** 2.66 ms (single-thread)
- **Checkpoint:** `C:\Users\MSchm\Documents\att_project\lightning_logs\M3_Medium_Baseline\version_0\checkpoints\M3_Medium_Baseline-epoch=34-val_loss=0.0013.ckpt`

### M4 Medium + Simple Attn

- **Parameters:** 597,762
- **FLOPs:** 30.138M
- **MACs:** 1
- **R2:** 0.9191
- **Accuracy:** 90.25%
- **RMSE:** 0.0311
- **MAE:** 0.0236
- **Inference (P95):** 2.75 ms (single-thread)
- **Checkpoint:** `C:\Users\MSchm\Documents\att_project\lightning_logs\M4_Medium_Simple_Attention\version_2\checkpoints\M4_Medium_Simple_Attention-epoch=40-val_loss=0.0011.ckpt`

### M5 Medium + Additive Attn

- **Parameters:** 630,529
- **FLOPs:** 31.770M
- **MACs:** 1
- **R2:** 0.9074
- **Accuracy:** 88.34%
- **RMSE:** 0.0332
- **MAE:** 0.0251
- **Inference (P95):** 3.82 ms (single-thread)
- **Checkpoint:** `C:\Users\MSchm\Documents\att_project\lightning_logs\M5_Medium_Additive_Attention\version_0\checkpoints\M5_Medium_Additive_Attention-epoch=32-val_loss=0.0012.ckpt`

### M6 Medium + Scaled DP

- **Parameters:** 597,633
- **FLOPs:** 30.131M
- **MACs:** 1
- **R2:** 0.9068
- **Accuracy:** 88.17%
- **RMSE:** 0.0334
- **MAE:** 0.0252
- **Inference (P95):** 2.77 ms (single-thread)
- **Checkpoint:** `C:\Users\MSchm\Documents\att_project\lightning_logs\M6_Medium_Scaled_DP_Attention\version_0\checkpoints\M6_Medium_Scaled_DP_Attention-epoch=27-val_loss=0.0012.ckpt`

---

*Results directory: `results/no_dropout/`*