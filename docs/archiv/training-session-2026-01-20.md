# Training Session 2026-01-20

## Status

Training läuft über Nacht (M4, M5, M6).

| Modell | Config | Status | R² | Accuracy |
|--------|--------|--------|-----|----------|
| M1 Small Baseline | m1_small_baseline.yaml | ✅ | 0.860 | 82.54% |
| M2 Small + Attention | m2_small_simple_attn.yaml | ✅ | 0.850 | 81.50% |
| M3 Medium Baseline | m3_medium_baseline.yaml | ✅ | 0.903 | 87.81% |
| M4 Medium + Simple | m4_medium_simple_attn.yaml | 🔄 Training | - | - |
| M5 Medium + Additive | m5_medium_additive_attn.yaml | ⏳ Queued | - | - |
| M6 Medium + Scaled DP | m6_medium_scaled_dp_attn.yaml | ⏳ Queued | - | - |

## Nach dem Training

### 1. Checkpoints finden

```powershell
Get-ChildItem -Recurse lightning_logs/M4*/version_0/checkpoints/*.ckpt
Get-ChildItem -Recurse lightning_logs/M5*/version_0/checkpoints/*.ckpt
Get-ChildItem -Recurse lightning_logs/M6*/version_0/checkpoints/*.ckpt
```

### 2. Evaluieren

Den besten Checkpoint (niedrigster val_loss) verwenden:

```powershell
python scripts/evaluate_model.py --checkpoint "lightning_logs/M4_Medium_Simple_Attention/version_0/checkpoints/M4_Medium_Simple_Attention-epoch=XX-val_loss=X.XXXX.ckpt" --config config/model_configs/m4_medium_simple_attn.yaml --output results/m4_results.json

python scripts/evaluate_model.py --checkpoint "lightning_logs/M5_Medium_Additive_Attention/version_0/checkpoints/M5_Medium_Additive_Attention-epoch=XX-val_loss=X.XXXX.ckpt" --config config/model_configs/m5_medium_additive_attn.yaml --output results/m5_results.json

python scripts/evaluate_model.py --checkpoint "lightning_logs/M6_Medium_Scaled_DP_Attention/version_0/checkpoints/M6_Medium_Scaled_DP_Attention-epoch=XX-val_loss=X.XXXX.ckpt" --config config/model_configs/m6_medium_scaled_dp_attn.yaml --output results/m6_results.json
```

### 3. Vergleichstabelle generieren

```powershell
python scripts/compare_results.py results/*.json --output docs/reports/comparison.md
python scripts/compare_results.py results/*.json --latex --output docs/reports/comparison.tex
```

### 4. Training Curves plotten

```powershell
python scripts/plot_training_curves.py --logdir lightning_logs/M4_Medium_Simple_Attention/version_0 --output results/figures/M4_Medium_Simple_Attention/
python scripts/plot_training_curves.py --logdir lightning_logs/M5_Medium_Additive_Attention/version_0 --output results/figures/M5_Medium_Additive_Attention/
python scripts/plot_training_curves.py --logdir lightning_logs/M6_Medium_Scaled_DP_Attention/version_0 --output results/figures/M6_Medium_Scaled_DP_Attention/
```

## Erwartete Ergebnisse

Laut Paper (Kapitel 5) sollte LSTM-Attention bei hidden=128, layers=5 besser abschneiden:
- Paper Baseline: 86.75%
- Paper Attention: 87.53%

Unsere M3 Baseline erreichte 87.81%, also sollten M4-M6 noch höher liegen.

## Erkenntnisse bisher

1. **Small Models (M1, M2):** Attention hilft nicht (bestätigt Paper)
2. **Medium Baseline (M3):** Deutliche Verbesserung durch mehr Parameter
3. **Plan-Änderung:** Configs wurden angepasst um Attention auf Medium-Größe zu testen

## Relevante Commits

- `8a7c8a6` - Config-Restructuring basierend auf Paper-Erkenntnissen
- `2188bf8` - M3 Ergebnisse
