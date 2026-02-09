# Model Evaluation Results - Dropout Ablation

> **Ziel:** Ablation Study: Einfluss von Dropout (0.2) auf alle Modellvarianten
>
> **Stand:** 2026-01-26

---

## Übersicht

| Modell | Typ | Parameter | R² | Accuracy | RMSE | Inference (ms) | Status |
|--------|-----|-----------|-----|----------|------|----------------|--------|
| M1 Small Baseline | LSTM (64, 3) | 84,801 | 0.844 | 80.49% | 0.0432 | 1.00 | ✅ |
| M2 Small + Simple Attn | LSTM + Attention (64, 3) | 84,866 | 0.834 | 80.07% | 0.0445 | 1.00 | ✅ |
| **M3 Medium Baseline** | LSTM (128, 5) | 597,633 | **0.893** | **86.29%** | **0.0357** | 2.09 | ✅ |
| M4 Medium + Simple Attn | LSTM + Attention (128, 5) | 597,762 | 0.877 | 84.31% | 0.0382 | 2.10 | ✅ |
| M5 Medium + Additive Attn | LSTM + Additive (128, 5) | 630,529 | 0.884 | 85.39% | 0.0372 | 2.52 | ✅ |
| M6 Medium + Scaled DP | LSTM + Scaled DP (128, 5) | 597,633 | 0.885 | 85.23% | 0.0370 | 2.26 | ✅ |

### Kernerkenntnisse

1. **Bestes Modell (mit Dropout):** M3 Medium Baseline (86.29% Accuracy, R²=0.893)
2. **Dropout schadet bei allen Modellen** - keine Verbesserung durch Regularisierung
3. **Small Models:** Dropout verstärkt Underfitting (-2.0% bis -2.5%)
4. **Medium Models mit Attention:** Negativer Effekt (-3.0% bis -5.9%)
5. **Ranking (Medium):** Baseline > Additive > Scaled DP > Simple

---

## Vergleich: Mit vs. Ohne Dropout

| Modell | Ohne Dropout | Mit Dropout | Δ Accuracy | Δ R² |
|--------|--------------|-------------|------------|------|
| M1 Small Baseline | 82.54% | 80.49% | **-2.05%** | -0.016 |
| M2 Small + Simple Attn | 81.91% | 80.07% | **-1.84%** | -0.020 |
| M3 Medium Baseline | 87.81% | 86.29% | **-1.52%** | -0.010 |
| M4 Medium + Simple Attn | 90.17% | 84.31% | **-5.86%** | -0.041 |
| M5 Medium + Additive Attn | 88.35% | 85.39% | **-2.96%** | -0.023 |
| M6 Medium + Scaled DP | 89.80% | 85.23% | **-4.57%** | -0.031 |

### Interpretation

- **Kein Overfitting vorhanden:** Die Modelle ohne Dropout zeigen kein Overfitting (Train/Val Loss parallel), daher ist Regularisierung kontraproduktiv
- **Attention-Mechanismen besonders betroffen:** M4 verliert fast 6% Accuracy durch Dropout
- **Empfehlung:** Für diese Aufgabe kein Dropout verwenden

---

## M1: Small Baseline LSTM (Dropout)

### Modell-Konfiguration

| Parameter | Wert |
|-----------|------|
| Typ | LSTM (baseline) |
| Input Size | 5 Features |
| Hidden Size | 64 |
| Num Layers | 3 |
| Dropout | **0.2** |
| **Total Parameters** | **84,801** |

**Relevante Dateien:**
- Config: [`config/model_configs/m1_small_baseline_dropout.yaml`](../../config/model_configs/m1_small_baseline_dropout.yaml)
- Checkpoint: [`lightning_logs/M1_Small_Baseline_Dropout/version_3/checkpoints/`](../../lightning_logs/M1_Small_Baseline_Dropout/version_3/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Vergleich zu Ohne Dropout |
|--------|------|---------------------------|
| **R²** | **0.8436** | -1.9% |
| **Accuracy** | **80.49%** | -2.5% |
| **RMSE** | 0.0432 | +5.9% |
| **MAE** | 0.0318 | +5.6% |
| **MSE** | 0.00187 | +12.0% |
| Test Samples | 220,127 | - |

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **1.00 ms** |
| Std | 0.28 ms |
| Min | 0.85 ms |
| Max | 7.79 ms |
| P50 | 0.92 ms |
| P95 | 1.21 ms |
| P99 | 1.61 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Results JSON: [`results/dropout/m1/eval.json`](../../results/dropout/m1/eval.json)
- Figures: [`results/dropout/m1/`](../../results/dropout/m1/)

---

## M2: Small + Simple Attention (Dropout)

### Modell-Konfiguration

| Parameter | Wert |
|-----------|------|
| Typ | LSTM + Simple Attention |
| Input Size | 5 Features |
| Hidden Size | 64 |
| Num Layers | 3 |
| Dropout | **0.2** |
| **Total Parameters** | **84,866** |

**Relevante Dateien:**
- Config: [`config/model_configs/m2_small_simple_attn_dropout.yaml`](../../config/model_configs/m2_small_simple_attn_dropout.yaml)
- Checkpoint: [`lightning_logs/M2_Small_Simple_Attention_Dropout/version_0/checkpoints/`](../../lightning_logs/M2_Small_Simple_Attention_Dropout/version_0/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Vergleich zu Ohne Dropout |
|--------|------|---------------------------|
| **R²** | **0.8341** | -2.3% |
| **Accuracy** | **80.07%** | -2.2% |
| **RMSE** | 0.0445 | +6.5% |
| **MAE** | 0.0323 | +4.5% |
| **MSE** | 0.00198 | +10.6% |
| Test Samples | 220,127 | - |

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **1.00 ms** |
| Std | 0.10 ms |
| Min | 0.88 ms |
| Max | 1.38 ms |
| P50 | 0.95 ms |
| P95 | 1.16 ms |
| P99 | 1.26 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Results JSON: [`results/dropout/m2/eval.json`](../../results/dropout/m2/eval.json)
- Figures: [`results/dropout/m2/`](../../results/dropout/m2/)

---

## M3: Medium Baseline (Dropout)

### Modell-Konfiguration

| Parameter | Wert |
|-----------|------|
| Typ | LSTM (baseline) |
| Input Size | 5 Features |
| Hidden Size | 128 |
| Num Layers | 5 |
| Dropout | **0.2** |
| **Total Parameters** | **597,633** |

**Relevante Dateien:**
- Config: [`config/model_configs/m3_medium_baseline_dropout.yaml`](../../config/model_configs/m3_medium_baseline_dropout.yaml)
- Checkpoint: [`lightning_logs/M3_Medium_Baseline_Dropout/version_0/checkpoints/`](../../lightning_logs/M3_Medium_Baseline_Dropout/version_0/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Vergleich zu Ohne Dropout |
|--------|------|---------------------------|
| **R²** | **0.8932** | -1.1% |
| **Accuracy** | **86.29%** | -1.7% |
| **RMSE** | 0.0357 | +5.0% |
| **MAE** | 0.0267 | +4.7% |
| **MSE** | 0.00128 | +10.3% |
| Test Samples | 220,127 | - |

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **2.09 ms** |
| Std | 0.20 ms |
| Min | 1.79 ms |
| Max | 2.73 ms |
| P50 | 2.00 ms |
| P95 | 2.42 ms |
| P99 | 2.57 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Results JSON: [`results/dropout/m3/eval.json`](../../results/dropout/m3/eval.json)
- Figures: [`results/dropout/m3/`](../../results/dropout/m3/)

---

## M4: Medium + Simple Attention (Dropout)

### Modell-Konfiguration

| Parameter | Wert |
|-----------|------|
| Typ | LSTM + Simple Attention |
| Input Size | 5 Features |
| Hidden Size | 128 |
| Num Layers | 5 |
| Dropout | **0.2** |
| **Total Parameters** | **597,762** |

**Relevante Dateien:**
- Config: [`config/model_configs/m4_medium_simple_attn_dropout.yaml`](../../config/model_configs/m4_medium_simple_attn_dropout.yaml)
- Checkpoint: [`lightning_logs/M4_Medium_Simple_Attention_Dropout/version_0/checkpoints/`](../../lightning_logs/M4_Medium_Simple_Attention_Dropout/version_0/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Vergleich zu Ohne Dropout |
|--------|------|---------------------------|
| **R²** | **0.8775** | **-4.4%** |
| **Accuracy** | **84.31%** | **-6.5%** |
| **RMSE** | 0.0382 | +22.0% |
| **MAE** | 0.0284 | +19.8% |
| **MSE** | 0.00146 | +49.0% |
| Test Samples | 220,127 | - |

> **Warnung:** Dropout hat bei M4 den stärksten negativen Effekt. Das Modell verliert fast 6% Accuracy.

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **2.10 ms** |
| Std | 0.20 ms |
| Min | 1.80 ms |
| Max | 2.96 ms |
| P50 | 2.00 ms |
| P95 | 2.46 ms |
| P99 | 2.55 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Results JSON: [`results/dropout/m4/eval.json`](../../results/dropout/m4/eval.json)
- Figures: [`results/dropout/m4/`](../../results/dropout/m4/)

---

## M5: Medium + Additive Attention (Dropout)

### Modell-Konfiguration

| Parameter | Wert |
|-----------|------|
| Typ | LSTM + Additive Attention (Bahdanau) |
| Input Size | 5 Features |
| Hidden Size | 128 |
| Num Layers | 5 |
| Dropout | **0.2** |
| **Total Parameters** | **630,529** |

**Relevante Dateien:**
- Config: [`config/model_configs/m5_medium_additive_attn_dropout.yaml`](../../config/model_configs/m5_medium_additive_attn_dropout.yaml)
- Checkpoint: [`lightning_logs/M5_Medium_Additive_Attention_Dropout/version_0/checkpoints/`](../../lightning_logs/M5_Medium_Additive_Attention_Dropout/version_0/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Vergleich zu Ohne Dropout |
|--------|------|---------------------------|
| **R²** | **0.8842** | -2.5% |
| **Accuracy** | **85.39%** | -3.4% |
| **RMSE** | 0.0372 | +11.7% |
| **MAE** | 0.0276 | +10.0% |
| **MSE** | 0.00138 | +24.3% |
| Test Samples | 220,127 | - |

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **2.52 ms** |
| Std | 0.22 ms |
| Min | 2.16 ms |
| Max | 4.58 ms |
| P50 | 2.46 ms |
| P95 | 2.91 ms |
| P99 | 3.03 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Results JSON: [`results/dropout/m5/eval.json`](../../results/dropout/m5/eval.json)
- Figures: [`results/dropout/m5/`](../../results/dropout/m5/)

---

## M6: Medium + Scaled Dot-Product Attention (Dropout)

### Modell-Konfiguration

| Parameter | Wert |
|-----------|------|
| Typ | LSTM + Scaled Dot-Product Attention |
| Input Size | 5 Features |
| Hidden Size | 128 |
| Num Layers | 5 |
| Dropout | **0.2** |
| **Total Parameters** | **597,633** |

**Relevante Dateien:**
- Config: [`config/model_configs/m6_medium_scaled_dp_attn_dropout.yaml`](../../config/model_configs/m6_medium_scaled_dp_attn_dropout.yaml)
- Checkpoint: [`lightning_logs/M6_Medium_Scaled_DP_Attention_Dropout/version_0/checkpoints/`](../../lightning_logs/M6_Medium_Scaled_DP_Attention_Dropout/version_0/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Vergleich zu Ohne Dropout |
|--------|------|---------------------------|
| **R²** | **0.8851** | **-3.4%** |
| **Accuracy** | **85.23%** | **-5.1%** |
| **RMSE** | 0.0370 | +16.7% |
| **MAE** | 0.0276 | +15.0% |
| **MSE** | 0.00137 | +37.0% |
| Test Samples | 220,127 | - |

> **Hinweis:** M6 zeigt moderaten Performance-Verlust durch Dropout, aber weniger stark als M4.

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **2.26 ms** |
| Std | 0.40 ms |
| Min | 1.80 ms |
| Max | 6.28 ms |
| P50 | 2.16 ms |
| P95 | 2.92 ms |
| P99 | 3.71 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Results JSON: [`results/dropout/m6/eval.json`](../../results/dropout/m6/eval.json)
- Figures: [`results/dropout/m6/`](../../results/dropout/m6/)

---

## Zusammenfassung für Paper

### Ablation Table (Dropout Effect)

| Model | Size | Dropout | Accuracy | R² | Δ Acc |
|-------|------|---------|----------|-----|-------|
| M1 Baseline | Small | 0.0 | **82.54%** | 0.860 | - |
| M1 Baseline | Small | 0.2 | 80.49% | 0.844 | -2.05% |
| M2 Simple Attn | Small | 0.0 | **81.91%** | 0.854 | - |
| M2 Simple Attn | Small | 0.2 | 80.07% | 0.834 | -1.84% |
| M3 Baseline | Medium | 0.0 | **87.81%** | 0.903 | - |
| M3 Baseline | Medium | 0.2 | 86.29% | 0.893 | -1.52% |
| M4 Simple Attn | Medium | 0.0 | **90.17%** | 0.918 | - |
| M4 Simple Attn | Medium | 0.2 | 84.31% | 0.877 | **-5.86%** |
| M5 Additive Attn | Medium | 0.0 | **88.35%** | 0.907 | - |
| M5 Additive Attn | Medium | 0.2 | 85.39% | 0.884 | -2.96% |
| M6 Scaled DP Attn | Medium | 0.0 | **89.80%** | 0.916 | - |
| M6 Scaled DP Attn | Medium | 0.2 | 85.23% | 0.885 | **-4.57%** |

### Paper-Formulierung (Entwurf)

> "We conducted an ablation study on dropout regularization (p=0.2) across all model variants.
> Results show that dropout consistently degrades performance for this task.
> Small models (~85K parameters) experienced accuracy drops of 1.8-2.1%, while medium models
> (~600K parameters) showed mixed effects: the baseline LSTM lost only 1.5%, but attention-augmented
> models suffered significant degradation (up to 5.9% for Simple Attention).
>
> This suggests that (1) our dataset does not induce overfitting in these model sizes, and
> (2) attention mechanisms are particularly sensitive to dropout regularization in the
> intermediate layers. All subsequent experiments use no dropout for optimal performance."

---

*Aktualisiert am: 2026-01-26*
