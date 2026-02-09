# Bootstrap Confidence Intervals (no_dropout)

Bootstrap samples: 1000

Uncertainty combines bootstrap sampling variance and between-seed variance (law of total variance).

| Model | Seeds | Accuracy (%) | RMSE | R² |
|-------|-------|-------------|------|-----|
| M1 MLP Last | 3 | 70.01 ± 0.10 | 0.0590 ± 0.0002 | 0.708 ± 0.003 |
| M2 MLP Flat | 3 | 74.90 ± 0.37 | 0.0509 ± 0.0004 | 0.783 ± 0.003 |
| M3 Small Baseline | 3 | 82.55 ± 0.18 | 0.0409 ± 0.0003 | 0.860 ± 0.003 |
| M4 Small + Simple Attn | 3 | 81.95 ± 0.26 | 0.0418 ± 0.0006 | 0.854 ± 0.004 |
| M5 Medium Baseline | 3 | 88.15 ± 0.80 | 0.0335 ± 0.0008 | 0.906 ± 0.005 |
| M6 Medium + Simple Attn | 3 | 90.04 ± 0.20 | 0.0314 ± 0.0003 | 0.917 ± 0.002 |
| M7 Medium + Additive Attn | 3 | 88.73 ± 0.35 | 0.0328 ± 0.0004 | 0.910 ± 0.002 |
| M8 Medium + Scaled DP | 3 | 88.62 ± 1.35 | 0.0330 ± 0.0016 | 0.909 ± 0.009 |

## Permutation Tests (10000 permutations)

| Comparison | Category | Δ Acc (%) | Δ RMSE | Δ R² | Cohen's d | Effect |
|------------|----------|-----------|--------|------|-----------|--------|
| M3 → M4 | Baseline vs Attention | -0.56 | +0.0008 | -0.005 | -0.053 | negligible |
| M5 → M6 | Baseline vs Attention | +1.87 | -0.0021 | +0.010 | +0.129 | negligible |
| M5 → M7 | Baseline vs Attention | +0.56 | -0.0006 | +0.003 | +0.039 | negligible |
| M5 → M8 | Baseline vs Attention | +0.73 | -0.0007 | +0.004 | +0.048 | negligible |
| M6 → M7 | Attention vs Attention | -1.31 | +0.0015 | -0.007 | -0.099 | negligible |
| M6 → M8 | Attention vs Attention | -1.14 | +0.0013 | -0.007 | -0.085 | negligible |
| M7 → M8 | Attention vs Attention | +0.16 | -0.0001 | +0.001 | +0.011 | negligible |
| M1 → M3 | MLP vs LSTM | +13.28 | -0.0191 | +0.157 | +0.386 | small |
| M2 → M5 | MLP vs LSTM | +14.49 | -0.0186 | +0.126 | +0.441 | small |

Cohen's d: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large