# Model Evaluation Results

> **Ziel:** Vergleich von LSTM-Architekturen mit verschiedenen Attention-Mechanismen für Steering Torque Prediction
>
> **Stand:** 2026-01-23

---

## Übersicht

| Modell | Typ | Parameter | R² | Accuracy | RMSE | Inference (ms) | Status |
|--------|-----|-----------|-----|----------|------|----------------|--------|
| M1 Small Baseline | LSTM (64, 3) | 84,801 | 0.860 | 82.54% | 0.0408 | 1.13 | ✅ |
| M2 Small + Simple Attn | LSTM + Attention (64, 3) | 84,866 | 0.854 | 81.91% | 0.0418 | 1.05 | ✅ |
| M3 Medium Baseline | LSTM (128, 5) | 597,633 | 0.903 | 87.81% | 0.0340 | 2.14 | ✅ |
| **M4 Medium + Simple Attn** | LSTM + Attention (128, 5) | 597,762 | **0.918** | **90.17%** | **0.0313** | 2.28 | ✅ |
| M5 Medium + Additive Attn | LSTM + Additive (128, 5) | 630,529 | 0.907 | 88.35% | 0.0333 | 2.64 | ✅ |
| M6 Medium + Scaled DP | LSTM + Scaled DP (128, 5) | 597,633 | 0.916 | 89.80% | 0.0317 | 2.27 | ✅ |

### Kernerkenntnisse

1. **Bestes Modell:** M4 Medium + Simple Attention (90.17% Accuracy, R²=0.918)
2. **Small Models:** Attention bringt keinen Vorteil (M2 < M1)
3. **Medium Models:** Alle Attention-Varianten übertreffen die Baseline
4. **Ranking (Medium):** Simple Attn > Scaled DP > Additive > Baseline
5. **Alle Modelle:** Erfüllen das Inferenz-Ziel (<10ms auf CPU)

---

## M1: Small Baseline LSTM

### Modell-Konfiguration

| Parameter | Wert |
|-----------|------|
| Typ | LSTM (baseline) |
| Input Size | 5 Features |
| Hidden Size | 64 |
| Num Layers | 3 |
| Output Size | 1 |
| **Total Parameters** | **84,801** |

**Relevante Dateien:**
- Config: [`config/model_configs/m1_small_baseline.yaml`](../../config/model_configs/m1_small_baseline.yaml)
- Model: [`model/LSTM.py`](../../model/LSTM.py)

### Training

| Parameter | Wert |
|-----------|------|
| Dataset | Paper (5001 files) |
| Samples | 2,201,265 |
| Train/Val/Test Split | 70/20/10 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Early Stopping | patience=5, monitor=val_loss |
| **Epochs trained** | **19** (Early Stop bei Epoch 14) |

**Relevante Dateien:**
- Training Script: [`scripts/train_model.py`](../../scripts/train_model.py)
- Base Config: [`config/base_config.yaml`](../../config/base_config.yaml)
- Checkpoint: [`lightning_logs/M1_Small_Baseline/version_0/checkpoints/M1_Small_Baseline-epoch=14-val_loss=0.0017.ckpt`](../../lightning_logs/M1_Small_Baseline/version_0/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Bemerkung |
|--------|------|-----------|
| **R²** | **0.8602** | Erklärt 86% der Varianz |
| **Accuracy** | **82.54%** | Threshold: ±0.05 |
| **RMSE** | 0.0408 | Root Mean Square Error |
| **MAE** | 0.0301 | Mean Absolute Error |
| **MSE** | 0.00167 | Mean Square Error |
| Test Samples | 220,127 | 10% des Datasets |

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **0.93 ms** |
| Std | 0.11 ms |
| Min | 0.81 ms |
| Max | 1.67 ms |
| P50 | 0.90 ms |
| P95 | 1.13 ms |
| P99 | 1.32 ms |
| **Target (<10 ms)** | **✅ PASS** |

> Die Inference-Zeit von <1ms ermöglicht problemlos 100Hz Echtzeit-Betrieb für EPS-Anwendungen.

**Relevante Dateien:**
- Evaluation Script: [`scripts/evaluate_model.py`](../../scripts/evaluate_model.py)
- Results JSON: [`results/m1_results.json`](../../results/m1_results.json)

### Trainingsmetriken (TensorBoard)

**TensorBoard starten:**
```bash
tensorboard --logdir lightning_logs/M1_Small_Baseline
```

**Logs:** [`lightning_logs/M1_Small_Baseline/version_0/`](../../lightning_logs/M1_Small_Baseline/version_0/)

---

## M2: Small + Simple Attention

### Modell-Konfiguration

| Parameter | Wert |
|-----------|------|
| Typ | LSTM + Simple Attention |
| Input Size | 5 Features |
| Hidden Size | 64 |
| Num Layers | 3 |
| Output Size | 1 |
| **Total Parameters** | **84,866** |

**Relevante Dateien:**
- Config: [`config/model_configs/m2_small_simple_attn.yaml`](../../config/model_configs/m2_small_simple_attn.yaml)
- Model: [`model/LSTM_attention.py`](../../model/LSTM_attention.py)

### Training

| Parameter | Wert |
|-----------|------|
| Dataset | Paper (5001 files) |
| Samples | 2,201,265 |
| Train/Val/Test Split | 70/20/10 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Early Stopping | patience=5, monitor=val_loss |
| **Epochs trained** | **17** (Early Stop bei Epoch 6) |

**Relevante Dateien:**
- Training Script: [`scripts/train_model.py`](../../scripts/train_model.py)
- Base Config: [`config/base_config.yaml`](../../config/base_config.yaml)
- Checkpoint: [`lightning_logs/M2_Small_Simple_Attention/version_0/checkpoints/M2_Small_Simple_Attention-epoch=06-val_loss=0.0018.ckpt`](../../lightning_logs/M2_Small_Simple_Attention/version_0/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Vergleich zu M1 |
|--------|------|-----------------|
| **R²** | **0.8503** | -1.15% |
| **Accuracy** | **81.50%** | -1.26% |
| **RMSE** | 0.0423 | +3.68% |
| **MAE** | 0.0309 | +2.66% |
| **MSE** | 0.00179 | +7.18% |
| Test Samples | 220,127 | - |

> **Hinweis:** Die Simple Attention (nur die letzte LSTM-Ausgabe mit Attention gewichten) bringt bei kleinen Modellen keinen Vorteil. Dies bestätigt die Ergebnisse aus dem Paper (Experiment 1).

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **1.02 ms** |
| Std | 0.13 ms |
| Min | 0.87 ms |
| Max | 1.78 ms |
| P50 | 0.97 ms |
| P95 | 1.25 ms |
| P99 | 1.39 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Evaluation Script: [`scripts/evaluate_model.py`](../../scripts/evaluate_model.py)
- Results JSON: [`results/m2_results.json`](../../results/m2_results.json)
- Training Curves: [`results/figures/M2_Small_Simple_Attention/training_curves.png/`](../../results/figures/M2_Small_Simple_Attention/training_curves.png/)

---

## M3: Medium Baseline LSTM

### Modell-Konfiguration

| Parameter | Wert |
|-----------|------|
| Typ | LSTM (baseline) |
| Input Size | 5 Features |
| Hidden Size | 128 |
| Num Layers | 5 |
| Output Size | 1 |
| **Total Parameters** | **597,633** |

**Relevante Dateien:**
- Config: [`config/model_configs/m3_medium_baseline.yaml`](../../config/model_configs/m3_medium_baseline.yaml)
- Model: [`model/LSTM.py`](../../model/LSTM.py)

### Training

| Parameter | Wert |
|-----------|------|
| Dataset | Paper (5001 files) |
| Samples | 2,201,265 |
| Train/Val/Test Split | 70/20/10 |
| Batch Size | 32 |
| Learning Rate | 0.0005 |
| Optimizer | Adam |
| Dropout | 0.0 |
| Early Stopping | patience=5, monitor=val_loss |
| **Epochs trained** | **44** (Early Stop bei Epoch 39) |

**Relevante Dateien:**
- Training Script: [`scripts/train_model.py`](../../scripts/train_model.py)
- Base Config: [`config/base_config.yaml`](../../config/base_config.yaml)
- Checkpoint: [`lightning_logs/M3_Medium_Baseline/version_0/checkpoints/M3_Medium_Baseline-epoch=39-val_loss=0.0013.ckpt`](../../lightning_logs/M3_Medium_Baseline/version_0/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Bemerkung |
|--------|------|-----------|
| **R²** | **0.9031** | Erklärt 90.3% der Varianz |
| **Accuracy** | **87.81%** | Threshold: ±0.05 |
| **RMSE** | 0.0340 | Root Mean Square Error |
| **MAE** | 0.0255 | Mean Absolute Error |
| **MSE** | 0.00116 | Mean Square Error |
| Test Samples | 220,127 | 10% des Datasets |

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **2.14 ms** |
| Std | 0.26 ms |
| Min | 1.74 ms |
| Max | 3.51 ms |
| P50 | 2.07 ms |
| P95 | 2.59 ms |
| P99 | 2.90 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Evaluation Script: [`scripts/evaluate_model.py`](../../scripts/evaluate_model.py)
- Results JSON: [`results/eval_m3_no_dropout.json`](../../results/eval_m3_no_dropout.json)

### Trainingsmetriken (TensorBoard)

**TensorBoard starten:**
```bash
tensorboard --logdir lightning_logs/M3_Medium_Baseline
```

**Logs:** [`lightning_logs/M3_Medium_Baseline/version_0/`](../../lightning_logs/M3_Medium_Baseline/version_0/)

---

## Daten

### Dataset

| Parameter | Wert |
|-----------|------|
| Vehicle | HYUNDAI_SONATA_2020 |
| Variant | paper (5001 files) |
| Window Size | 50 (5 Sekunden @ 10Hz) |
| Predict Size | 1 |
| Step Size | 1 |

### Features (Input)

| # | Feature | Beschreibung |
|---|---------|--------------|
| 1 | vEgo | Fahrzeuggeschwindigkeit (m/s) |
| 2 | aEgo | Längsbeschleunigung (m/s²) |
| 3 | steeringAngleDeg | Lenkradwinkel (Grad) |
| 4 | roll | Straßen-Rollwinkel (rad) |
| 5 | latAccelLocalizer | Querbeschleunigung (m/s²) |

### Target (Output)

| Feature | Beschreibung | Bereich |
|---------|--------------|---------|
| steerFiltered | Normiertes Lenkmoment | [-1, 1] |

**Relevante Dateien:**
- Data Module: [`data_module.py`](../../data_module.py)
- Config: [`config.py`](../../config.py)
- Preprocessed Data: `data/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF/`

---

## Metriken-Definitionen

| Metrik | Formel | Beschreibung |
|--------|--------|--------------|
| MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ | Mean Squared Error |
| RMSE | $\sqrt{MSE}$ | Root Mean Squared Error |
| MAE | $\frac{1}{n}\sum|y - \hat{y}|$ | Mean Absolute Error |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Bestimmtheitsmaß |
| Accuracy | $\frac{|\hat{y} - y| < threshold}{n}$ | Anteil innerhalb Threshold |

> **Hinweis:** Die R²-Berechnung während des Trainings (avg_val_r2) ist fehlerhaft aggregiert. Die korrekte R²-Berechnung erfolgt im Evaluation-Script über den gesamten Test-Set.

---

## Vergleichstabelle generieren

Nach Abschluss aller Modelle:

```bash
# Markdown-Tabelle
python scripts/compare_results.py results/*.json --output docs/reports/comparison.md

# LaTeX-Tabelle für Paper
python scripts/compare_results.py results/*.json --latex --output docs/reports/comparison.tex
```

**Script:** [`scripts/compare_results.py`](../../scripts/compare_results.py)

---

## Nächste Schritte

1. [x] M1 Training (Small Baseline) - ✅ R²=0.860, Acc=82.54%
2. [x] M2 Training (Small + Simple Attention) - ✅ R²=0.854, Acc=81.91%
   - Bestätigt: Attention hilft nicht bei kleinen Modellen
3. [x] M3 Training (Medium Baseline) - ✅ R²=0.903, Acc=87.81%
4. [x] M4 Training (Medium + Simple Attention) - ✅ R²=0.918, Acc=90.17% **BEST**
5. [x] M5 Training (Medium + Additive Attention) - ✅ R²=0.907, Acc=88.35%
6. [x] M6 Training (Medium + Scaled Dot-Product) - ✅ R²=0.916, Acc=89.80%
7. [ ] Dropout-Ablation (M1-M6 mit dropout=0.2)
8. [ ] Vergleichstabellen generieren
9. [ ] Attention-Visualisierung für Paper erstellen

---

## M4: Medium + Simple Attention

### Modell-Konfiguration

| Parameter | Wert |
|-----------|------|
| Typ | LSTM + Simple Attention |
| Input Size | 5 Features |
| Hidden Size | 128 |
| Num Layers | 5 |
| Output Size | 1 |
| Attention | Simple (score_i = W * h_i + b) |
| **Total Parameters** | **597,762** |

**Relevante Dateien:**
- Config: [`config/model_configs/m4_medium_simple_attn.yaml`](../../config/model_configs/m4_medium_simple_attn.yaml)
- Model: [`model/LSTM_attention.py`](../../model/LSTM_attention.py)

### Training

| Parameter | Wert |
|-----------|------|
| Dataset | Paper (5001 files) |
| Samples | 2,201,265 |
| Train/Val/Test Split | 70/20/10 |
| Batch Size | 32 |
| Learning Rate | 0.0005 |
| Optimizer | Adam |
| Dropout | 0.0 |
| Early Stopping | patience=5, monitor=val_loss |
| **Epochs trained** | **48** (Early Stop bei Epoch 43) |

**Relevante Dateien:**
- Training Script: [`scripts/train_model.py`](../../scripts/train_model.py)
- Base Config: [`config/base_config.yaml`](../../config/base_config.yaml)
- Checkpoint: [`lightning_logs/M4_Medium_Simple_Attention/version_2/checkpoints/M4_Medium_Simple_Attention-epoch=43-val_loss=0.0011.ckpt`](../../lightning_logs/M4_Medium_Simple_Attention/version_2/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Vergleich zu M3 (Baseline) |
|--------|------|----------------------------|
| **R²** | **0.9181** | +1.66% |
| **Accuracy** | **90.17%** | +2.69% |
| **RMSE** | 0.0313 | -7.94% |
| **MAE** | 0.0237 | -7.11% |
| **MSE** | 0.00098 | -15.52% |
| Test Samples | 220,127 | - |

> **Bestes Modell:** M4 erreicht die höchste Accuracy (90.17%) und R² (0.918) aller getesteten Modelle.

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **2.28 ms** |
| Std | 0.51 ms |
| Min | 1.84 ms |
| Max | 9.31 ms |
| P50 | 2.23 ms |
| P95 | 2.73 ms |
| P99 | 4.04 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Evaluation Script: [`scripts/evaluate_model.py`](../../scripts/evaluate_model.py)
- Results JSON: [`results/eval_m4_no_dropout.json`](../../results/eval_m4_no_dropout.json)
- Attention Weights: [`results/figures/M4_Medium_Simple_Attention/M4_Medium_Simple_Attention_attention_weights.npy`](../../results/figures/M4_Medium_Simple_Attention/)

### Trainingsmetriken (TensorBoard)

**TensorBoard starten:**
```bash
tensorboard --logdir lightning_logs/M4_Medium_Simple_Attention
```

**Logs:** [`lightning_logs/M4_Medium_Simple_Attention/version_2/`](../../lightning_logs/M4_Medium_Simple_Attention/version_2/)

---

## M5: Medium + Additive Attention (Bahdanau)

### Modell-Konfiguration

| Parameter | Wert |
|-----------|------|
| Typ | LSTM + Additive Attention |
| Input Size | 5 Features |
| Hidden Size | 128 |
| Num Layers | 5 |
| Output Size | 1 |
| Attention | Bahdanau (score_ij = v^T * tanh(W * h_i + U * h_j)) |
| Attention Size | 128 |
| **Total Parameters** | **630,529** |

**Relevante Dateien:**
- Config: [`config/model_configs/m5_medium_additive_attn.yaml`](../../config/model_configs/m5_medium_additive_attn.yaml)
- Model: [`model/LSTM_attention.py`](../../model/LSTM_attention.py)

### Training

| Parameter | Wert |
|-----------|------|
| Dataset | Paper (5001 files) |
| Samples | 2,201,265 |
| Train/Val/Test Split | 70/20/10 |
| Batch Size | 32 |
| Learning Rate | 0.0005 |
| Optimizer | Adam |
| Dropout | 0.0 |
| Early Stopping | patience=5, monitor=val_loss |
| **Epochs trained** | **38** (Early Stop bei Epoch 33) |

**Relevante Dateien:**
- Training Script: [`scripts/train_model.py`](../../scripts/train_model.py)
- Base Config: [`config/base_config.yaml`](../../config/base_config.yaml)
- Checkpoint: [`lightning_logs/M5_Medium_Additive_Attention/version_0/checkpoints/M5_Medium_Additive_Attention-epoch=33-val_loss=0.0012.ckpt`](../../lightning_logs/M5_Medium_Additive_Attention/version_0/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Vergleich zu M3 (Baseline) |
|--------|------|----------------------------|
| **R²** | **0.9072** | +0.45% |
| **Accuracy** | **88.35%** | +0.61% |
| **RMSE** | 0.0333 | -2.06% |
| **MAE** | 0.0251 | -1.65% |
| **MSE** | 0.00111 | -4.31% |
| Test Samples | 220,127 | - |

> **Hinweis:** Die Additive Attention hat die meisten Parameter (+5.5% vs M3), aber nicht die beste Performance. Simple Attention ist effizienter.

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **2.64 ms** |
| Std | 0.36 ms |
| Min | 2.14 ms |
| Max | 6.32 ms |
| P50 | 2.62 ms |
| P95 | 3.26 ms |
| P99 | 3.57 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Evaluation Script: [`scripts/evaluate_model.py`](../../scripts/evaluate_model.py)
- Results JSON: [`results/eval_m5_no_dropout.json`](../../results/eval_m5_no_dropout.json)
- Attention Weights: [`results/figures/M5_Medium_Additive_Attention/M5_Medium_Additive_Attention_attention_weights.npy`](../../results/figures/M5_Medium_Additive_Attention/)

### Trainingsmetriken (TensorBoard)

**TensorBoard starten:**
```bash
tensorboard --logdir lightning_logs/M5_Medium_Additive_Attention
```

**Logs:** [`lightning_logs/M5_Medium_Additive_Attention/version_0/`](../../lightning_logs/M5_Medium_Additive_Attention/version_0/)

---

## M6: Medium + Scaled Dot-Product Attention

### Modell-Konfiguration

| Parameter | Wert |
|-----------|------|
| Typ | LSTM + Scaled Dot-Product Attention |
| Input Size | 5 Features |
| Hidden Size | 128 |
| Num Layers | 5 |
| Output Size | 1 |
| Attention | Transformer-style (score_ij = (h_i · h_j) / sqrt(d)) |
| **Total Parameters** | **597,633** |

**Relevante Dateien:**
- Config: [`config/model_configs/m6_medium_scaled_dp_attn.yaml`](../../config/model_configs/m6_medium_scaled_dp_attn.yaml)
- Model: [`model/LSTM_attention.py`](../../model/LSTM_attention.py)

### Training

| Parameter | Wert |
|-----------|------|
| Dataset | Paper (5001 files) |
| Samples | 2,201,265 |
| Train/Val/Test Split | 70/20/10 |
| Batch Size | 32 |
| Learning Rate | 0.0005 |
| Optimizer | Adam |
| Dropout | 0.0 |
| Early Stopping | patience=5, monitor=val_loss |
| **Epochs trained** | **47** (Early Stop bei Epoch 42) |

**Relevante Dateien:**
- Training Script: [`scripts/train_model.py`](../../scripts/train_model.py)
- Base Config: [`config/base_config.yaml`](../../config/base_config.yaml)
- Checkpoint: [`lightning_logs/M6_Medium_Scaled_DP_Attention/version_1/checkpoints/M6_Medium_Scaled_DP_Attention-epoch=42-val_loss=0.0012.ckpt`](../../lightning_logs/M6_Medium_Scaled_DP_Attention/version_1/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Vergleich zu M3 (Baseline) |
|--------|------|----------------------------|
| **R²** | **0.9161** | +1.44% |
| **Accuracy** | **89.80%** | +2.27% |
| **RMSE** | 0.0317 | -6.76% |
| **MAE** | 0.0240 | -5.82% |
| **MSE** | 0.00100 | -13.79% |
| Test Samples | 220,127 | - |

> **Hinweis:** Scaled Dot-Product (Transformer-Style) ist der zweitbeste Attention-Mechanismus, nur knapp hinter Simple Attention.

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **2.27 ms** |
| Std | 0.25 ms |
| Min | 1.82 ms |
| Max | 3.80 ms |
| P50 | 2.30 ms |
| P95 | 2.70 ms |
| P99 | 2.95 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Evaluation Script: [`scripts/evaluate_model.py`](../../scripts/evaluate_model.py)
- Results JSON: [`results/eval_m6_no_dropout.json`](../../results/eval_m6_no_dropout.json)
- Attention Weights: [`results/figures/M6_Medium_Scaled_DP_Attention/M6_Medium_Scaled_DP_Attention_attention_weights.npy`](../../results/figures/M6_Medium_Scaled_DP_Attention/)

### Trainingsmetriken (TensorBoard)

**TensorBoard starten:**
```bash
tensorboard --logdir lightning_logs/M6_Medium_Scaled_DP_Attention
```

**Logs:** [`lightning_logs/M6_Medium_Scaled_DP_Attention/version_1/`](../../lightning_logs/M6_Medium_Scaled_DP_Attention/version_1/)

---

*Aktualisiert am: 2026-01-26*
