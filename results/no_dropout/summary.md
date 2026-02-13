# Evaluation Results - No Dropout

> Generated: 2026-02-13 11:03:58
> Results show mean +/- std across multiple seeds

## Overview

| Model | Type | Params | R2 | Accuracy | RMSE | Seeds |
|-------|------|--------|-----|----------|------|-------|
| M1 MLP Last | MLP (5→64→64→1) | 4,609 | 0.692 +/- 0.006 | 70.1% +/- 0.3 | 0.0591 +/- 0.0006 | 5 |
| M2 MLP Flat | MLP (250→128→64→1) | 40,449 | 0.771 +/- 0.005 | 74.3% +/- 0.5 | 0.0510 +/- 0.0006 | 5 |
| M3 Small Baseline | LSTM (64, 3) | 84,801 | 0.826 +/- 0.002 | 80.3% +/- 0.4 | 0.0444 +/- 0.0003 | 5 |
| M4 Small + Simple Attn | LSTM + Attention (64, 3) | 84,866 | 0.828 +/- 0.001 | 80.3% +/- 0.1 | 0.0443 +/- 0.0002 | 5 |
| M5 Medium Baseline | LSTM (128, 5) | 597,633 | 0.828 +/- 0.003 | 80.5% +/- 0.3 | 0.0442 +/- 0.0004 | 5 |
| **M6 Medium + Simple Attn** | LSTM + Attention (128, 5) | 597,762 | **0.831 +/- 0.003** | **80.5% +/- 0.3** | 0.0438 +/- 0.0004 | 5 |
| M7 Medium + Additive Attn | LSTM + Additive (128, 5) | 630,529 | 0.830 +/- 0.002 | 80.6% +/- 0.2 | 0.0439 +/- 0.0002 | 5 |
| M8 Medium + Scaled DP | LSTM + Scaled DP (128, 5) | 597,633 | 0.825 +/- 0.002 | 80.2% +/- 0.2 | 0.0445 +/- 0.0003 | 5 |

## Key Findings

- **Best Model:** M6 Medium + Simple Attn (R2=0.831)
- **Variant:** No Dropout
- **Models Evaluated:** 8/8
- **Note:** Results aggregated across multiple random seeds

## Model Details

### M1 MLP Last

- **Parameters:** 4,609
- **FLOPs:** 4.480K
- **MACs:** 2
- **R2:** 0.6924 +/- 0.0064
- **Accuracy:** 70.11% +/- 0.26
- **RMSE:** 0.0591 +/- 0.0006
- **MAE:** 0.0422 +/- 0.0003
- **Inference (P95):** 0.07 +/- 0.00 ms
- **Seeds:** [7, 42, 94, 123, 231]

### M2 MLP Flat

- **Parameters:** 40,449
- **FLOPs:** 40.256K
- **MACs:** 2
- **R2:** 0.7708 +/- 0.0053
- **Accuracy:** 74.35% +/- 0.51
- **RMSE:** 0.0510 +/- 0.0006
- **MAE:** 0.0373 +/- 0.0004
- **Inference (P95):** 0.06 +/- 0.00 ms
- **Seeds:** [7, 42, 94, 123, 231]

### M3 Small Baseline

- **Parameters:** 84,801
- **FLOPs:** 4.314M
- **MACs:** 2
- **R2:** 0.8265 +/- 0.0021
- **Accuracy:** 80.34% +/- 0.37
- **RMSE:** 0.0444 +/- 0.0003
- **MAE:** 0.0321 +/- 0.0003
- **Inference (P95):** 0.78 +/- 0.02 ms
- **Seeds:** [7, 42, 94, 123, 231]

### M4 Small + Simple Attn

- **Parameters:** 84,866
- **FLOPs:** 4.317M
- **MACs:** 2
- **R2:** 0.8275 +/- 0.0012
- **Accuracy:** 80.28% +/- 0.15
- **RMSE:** 0.0443 +/- 0.0002
- **MAE:** 0.0321 +/- 0.0001
- **Inference (P95):** 0.81 +/- 0.02 ms
- **Seeds:** [7, 42, 94, 123, 231]

### M5 Medium Baseline

- **Parameters:** 597,633
- **FLOPs:** 30.131M
- **MACs:** 1
- **R2:** 0.8279 +/- 0.0027
- **Accuracy:** 80.55% +/- 0.30
- **RMSE:** 0.0442 +/- 0.0004
- **MAE:** 0.0319 +/- 0.0002
- **Inference (P95):** 2.62 +/- 0.01 ms
- **Seeds:** [7, 42, 94, 123, 231]

### M6 Medium + Simple Attn

- **Parameters:** 597,762
- **FLOPs:** 30.138M
- **MACs:** 1
- **R2:** 0.8308 +/- 0.0027
- **Accuracy:** 80.49% +/- 0.31
- **RMSE:** 0.0438 +/- 0.0004
- **MAE:** 0.0318 +/- 0.0003
- **Inference (P95):** 2.68 +/- 0.01 ms
- **Seeds:** [7, 42, 94, 123, 231]

### M7 Medium + Additive Attn

- **Parameters:** 630,529
- **FLOPs:** 31.770M
- **MACs:** 1
- **R2:** 0.8305 +/- 0.0019
- **Accuracy:** 80.56% +/- 0.23
- **RMSE:** 0.0439 +/- 0.0002
- **MAE:** 0.0318 +/- 0.0002
- **Inference (P95):** 3.82 +/- 0.02 ms
- **Seeds:** [7, 42, 94, 123, 231]

### M8 Medium + Scaled DP

- **Parameters:** 597,633
- **FLOPs:** 30.131M
- **MACs:** 1
- **R2:** 0.8254 +/- 0.0021
- **Accuracy:** 80.17% +/- 0.20
- **RMSE:** 0.0445 +/- 0.0003
- **MAE:** 0.0322 +/- 0.0002
- **Inference (P95):** 3.24 +/- 0.28 ms
- **Seeds:** [7, 42, 94, 123, 231]

---

*Results directory: `results/no_dropout/`*