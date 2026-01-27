# Bootstrap Confidence Intervals (no_dropout)

Bootstrap samples: 1000

| Model | Accuracy (%) | RMSE | R² |
|-------|-------------|------|-----|
| M1 Small Baseline | 82.58 ± 0.08 | 0.0408 ± 0.0001 | 0.860 ± 0.001 |
| M2 Small + Simple Attn | 81.50 ± 0.09 | 0.0423 ± 0.0001 | 0.850 ± 0.001 |
| M3 Medium Baseline | 87.84 ± 0.07 | 0.0338 ± 0.0001 | 0.905 ± 0.001 |
| M4 Medium + Simple Attn | 90.25 ± 0.07 | 0.0311 ± 0.0001 | 0.919 ± 0.001 |
| M5 Medium + Additive Attn | 88.34 ± 0.07 | 0.0332 ± 0.0001 | 0.907 ± 0.001 |
| M6 Medium + Scaled DP | 89.60 ± 0.07 | 0.0319 ± 0.0001 | 0.915 ± 0.001 |