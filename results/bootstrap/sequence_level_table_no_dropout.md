# Sequence-Level Bootstrap CIs (no_dropout)

Bootstrap samples: 1000  
Test sequences: 500  
Accuracy threshold: 0.05

Uncertainty combines bootstrap sampling variance and between-seed variance (law of total variance).

| Model | Seeds | Seqs | Accuracy (%) | RMSE | MAE |
|-------|-------|------|-------------|------|-----|
| M1 MLP Last | 5 | 500 | 68.98 ± 0.82 | 0.0559 ± 0.0012 | 0.0434 ± 0.0009 |
| M2 MLP Flat | 5 | 500 | 73.44 ± 0.86 | 0.0492 ± 0.0011 | 0.0384 ± 0.0008 |
| M3 Small Baseline | 5 | 500 | 79.46 ± 0.73 | 0.0425 ± 0.0008 | 0.0328 ± 0.0006 |
| M4 Small + Simple Attn | 5 | 500 | 79.41 ± 0.63 | 0.0425 ± 0.0007 | 0.0328 ± 0.0006 |
| M5 Medium Baseline | 5 | 500 | 79.63 ± 0.68 | 0.0423 ± 0.0008 | 0.0327 ± 0.0006 |
| M6 Medium + Simple Attn | 5 | 500 | 79.60 ± 0.69 | 0.0422 ± 0.0008 | 0.0326 ± 0.0006 |
| M7 Medium + Additive Attn | 5 | 500 | 79.73 ± 0.64 | 0.0421 ± 0.0008 | 0.0325 ± 0.0006 |
| M8 Medium + Scaled DP | 5 | 500 | 79.26 ± 0.63 | 0.0427 ± 0.0008 | 0.0330 ± 0.0006 |

## Sequence-Level Permutation Tests (10000 permutations)

| Comparison | Category | Seqs | Δ Acc (%) | Δ RMSE | Δ MAE | Cohen's d | Hedge's g | Effect |
|------------|----------|------|-----------|--------|------|-----------|-----------|--------|
| M3 → M4 | Baseline vs Attention | 500 | -0.05 | -0.0000 | -0.0000 | +0.006 | +0.005 | negligible |
| M5 → M6 | Baseline vs Attention | 500 | -0.03 | -0.0002 | -0.0001 | +0.035 | +0.035 | negligible |
| M5 → M7 | Baseline vs Attention | 500 | +0.11 | -0.0002 | -0.0001 | +0.090 | +0.090 | negligible |
| M5 → M8 | Baseline vs Attention | 500 | -0.37*** | +0.0004 | +0.0004 | -0.176 | -0.176 | negligible |
| M6 → M7 | Attention vs Attention | 500 | +0.13 | -0.0000 | -0.0001 | +0.047 | +0.047 | negligible |
| M6 → M8 | Attention vs Attention | 500 | -0.34*** | +0.0006 | +0.0004 | -0.252 | -0.252 | small |
| M7 → M8 | Attention vs Attention | 500 | -0.48*** | +0.0006 | +0.0005 | -0.303 | -0.303 | small |
| M1 → M3 | MLP vs LSTM | 500 | +10.48*** | -0.0134 | -0.0106 | +1.081 | +1.079 | large |
| M2 → M5 | MLP vs LSTM | 500 | +6.19*** | -0.0069 | -0.0057 | +0.691 | +0.690 | medium |

Cohen's d: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large
Significance: * p<0.05, ** p<0.01, *** p<0.001