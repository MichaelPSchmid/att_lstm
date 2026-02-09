# Evaluation Results - No Dropout

> Generated: 2026-02-09 09:18:03
> Results show mean +/- std across multiple seeds

## Overview

| Model | Type | Params | R2 | Accuracy | RMSE | Seeds |
|-------|------|--------|-----|----------|------|-------|
| M1 MLP Last | MLP (5→64→64→1) | 4,609 | 0.708 +/- 0.001 | 70.0% +/- 0.0 | 0.0590 +/- 0.0001 | 3 |
| M2 MLP Flat | MLP (250→128→64→1) | 40,449 | 0.783 +/- 0.002 | 74.9% +/- 0.3 | 0.0509 +/- 0.0003 | 3 |
| M3 Small Baseline | LSTM (64, 3) | 84,801 | 0.860 +/- 0.002 | 82.5% +/- 0.1 | 0.0409 +/- 0.0003 | 3 |
| M4 Small + Simple Attn | LSTM + Attention (64, 3) | 84,866 | 0.854 +/- 0.003 | 82.0% +/- 0.2 | 0.0418 +/- 0.0005 | 3 |
| M5 Medium Baseline | LSTM (128, 5) | 597,633 | 0.906 +/- 0.004 | 88.2% +/- 0.7 | 0.0335 +/- 0.0007 | 3 |
| **M6 Medium + Simple Attn** | LSTM + Attention (128, 5) | 597,762 | **0.917 +/- 0.001** | **90.0% +/- 0.2** | 0.0314 +/- 0.0002 | 3 |
| M7 Medium + Additive Attn | LSTM + Additive (128, 5) | 630,529 | 0.910 +/- 0.002 | 88.7% +/- 0.3 | 0.0328 +/- 0.0003 | 3 |
| M8 Medium + Scaled DP | LSTM + Scaled DP (128, 5) | 597,633 | 0.909 +/- 0.007 | 88.6% +/- 1.1 | 0.0330 +/- 0.0013 | 3 |

## Key Findings

- **Best Model:** M6 Medium + Simple Attn (R2=0.917)
- **Variant:** No Dropout
- **Models Evaluated:** 8/8
- **Note:** Results aggregated across multiple random seeds

## Model Details

### M1 MLP Last

- **Parameters:** 4,609
- **FLOPs:** 4.480K
- **MACs:** 2
- **R2:** 0.7084 +/- 0.0013
- **Accuracy:** 70.01% +/- 0.03
- **RMSE:** 0.0590 +/- 0.0001
- **MAE:** 0.0422 +/- 0.0001
- **Inference (P95):** 0.07 +/- 0.00 ms
- **Seeds:** [42, 94, 123]

### M2 MLP Flat

- **Parameters:** 40,449
- **FLOPs:** 40.256K
- **MACs:** 2
- **R2:** 0.7830 +/- 0.0023
- **Accuracy:** 74.90% +/- 0.29
- **RMSE:** 0.0509 +/- 0.0003
- **MAE:** 0.0369 +/- 0.0002
- **Inference (P95):** 0.06 +/- 0.00 ms
- **Seeds:** [42, 94, 123]

### M3 Small Baseline

- **Parameters:** 84,801
- **FLOPs:** 4.314M
- **MACs:** 2
- **R2:** 0.8595 +/- 0.0019
- **Accuracy:** 82.55% +/- 0.13
- **RMSE:** 0.0409 +/- 0.0003
- **MAE:** 0.0300 +/- 0.0001
- **Inference (P95):** 0.78 +/- 0.01 ms
- **Seeds:** [42, 94, 123]

### M4 Small + Simple Attn

- **Parameters:** 84,866
- **FLOPs:** 4.317M
- **MACs:** 2
- **R2:** 0.8536 +/- 0.0033
- **Accuracy:** 81.95% +/- 0.21
- **RMSE:** 0.0418 +/- 0.0005
- **MAE:** 0.0305 +/- 0.0002
- **Inference (P95):** 0.83 +/- 0.02 ms
- **Seeds:** [42, 94, 123]

### M5 Medium Baseline

- **Parameters:** 597,633
- **FLOPs:** 30.131M
- **MACs:** 1
- **R2:** 0.9062 +/- 0.0037
- **Accuracy:** 88.15% +/- 0.65
- **RMSE:** 0.0335 +/- 0.0007
- **MAE:** 0.0253 +/- 0.0005
- **Inference (P95):** 2.66 +/- 0.03 ms
- **Seeds:** [42, 94, 123]

### M6 Medium + Simple Attn

- **Parameters:** 597,762
- **FLOPs:** 30.138M
- **MACs:** 1
- **R2:** 0.9174 +/- 0.0012
- **Accuracy:** 90.04% +/- 0.16
- **RMSE:** 0.0314 +/- 0.0002
- **MAE:** 0.0238 +/- 0.0001
- **Inference (P95):** 2.72 +/- 0.03 ms
- **Seeds:** [42, 94, 123]

### M7 Medium + Additive Attn

- **Parameters:** 630,529
- **FLOPs:** 31.770M
- **MACs:** 1
- **R2:** 0.9096 +/- 0.0016
- **Accuracy:** 88.73% +/- 0.28
- **RMSE:** 0.0328 +/- 0.0003
- **MAE:** 0.0248 +/- 0.0002
- **Inference (P95):** 3.90 +/- 0.04 ms
- **Seeds:** [42, 94, 123]

### M8 Medium + Scaled DP

- **Parameters:** 597,633
- **FLOPs:** 30.131M
- **MACs:** 1
- **R2:** 0.9087 +/- 0.0075
- **Accuracy:** 88.62% +/- 1.10
- **RMSE:** 0.0330 +/- 0.0013
- **MAE:** 0.0249 +/- 0.0009
- **Inference (P95):** 2.78 +/- 0.01 ms
- **Seeds:** [42, 94, 123]

---

*Results directory: `results/no_dropout/`*