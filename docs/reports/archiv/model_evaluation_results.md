# Model Evaluation Results

> **Ziel:** Vergleich von LSTM-Architekturen mit verschiedenen Attention-Mechanismen für Steering Torque Prediction
>
> **Stand:** 2026-01-23

---

## Übersicht

| Modell | Typ | Parameter | R² | Accuracy | RMSE | Inference (ms) | Status |
|--------|-----|-----------|-----|----------|------|----------------|--------|
| M1 Small Baseline | LSTM (64, 3) | 84,801 | 0.860 | 82.57% | 0.0408 | 1.11 | ✅ |
| M2 Small + Simple Attn | LSTM + Attention (64, 3) | 84,866 | 0.850 | 81.50% | 0.0423 | 1.16 | ✅ |
| M3 Medium Baseline | LSTM (128, 5) | 597,633 | 0.905 | 87.84% | 0.0338 | 2.40 | ✅ |
| **M4 Medium + Simple Attn** | LSTM + Attention (128, 5) | 597,762 | **0.919** | **90.25%** | **0.0311** | 2.44 | ✅ |
| M5 Medium + Additive Attn | LSTM + Additive (128, 5) | 630,529 | 0.907 | 88.34% | 0.0332 | 2.88 | ✅ |
| M6 Medium + Scaled DP | LSTM + Scaled DP (128, 5) | 597,633 | 0.907 | 88.17% | 0.0334 | 2.46 | ✅ |

### Kernerkenntnisse

1. **Bestes Modell:** M4 Medium + Simple Attention (90.25% Accuracy, R²=0.919)
2. **Small Models:** Attention bringt keinen Vorteil (M2 < M1)
3. **Medium Models:** Alle Attention-Varianten übertreffen die Baseline
4. **Ranking (Medium):** Simple Attn > Additive ≈ Scaled DP > Baseline
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
| **Epochs trained** | **17** (Early Stop bei Epoch 12) |

**Relevante Dateien:**
- Training Script: [`scripts/train_model.py`](../../scripts/train_model.py)
- Base Config: [`config/base_config.yaml`](../../config/base_config.yaml)
- Checkpoint: [`lightning_logs/M1_Small_Baseline/version_0/checkpoints/M1_Small_Baseline-epoch=12-val_loss=0.0017.ckpt`](../../lightning_logs/M1_Small_Baseline/version_0/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Bemerkung |
|--------|------|-----------|
| **R²** | **0.8604** | Erklärt 86% der Varianz |
| **Accuracy** | **82.57%** | Threshold: ±0.05 |
| **RMSE** | 0.0408 | Root Mean Square Error |
| **MAE** | 0.0300 | Mean Absolute Error |
| **MSE** | 0.00167 | Mean Square Error |
| Test Samples | 220,127 | 10% des Datasets |

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **0.94 ms** |
| Std | 0.10 ms |
| Min | 0.83 ms |
| Max | 1.61 ms |
| P50 | 0.89 ms |
| P95 | 1.11 ms |
| P99 | 1.20 ms |
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
| **R²** | **0.8503** | -1.18% |
| **Accuracy** | **81.50%** | -1.30% |
| **RMSE** | 0.0423 | +3.58% |
| **MAE** | 0.0309 | +3.00% |
| **MSE** | 0.00179 | +7.27% |
| Test Samples | 220,127 | - |

> **Hinweis:** Die Simple Attention (nur die letzte LSTM-Ausgabe mit Attention gewichten) bringt bei kleinen Modellen keinen Vorteil. Dies bestätigt die Ergebnisse aus dem Paper (Experiment 1).

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **1.00 ms** |
| Std | 0.11 ms |
| Min | 0.87 ms |
| Max | 1.58 ms |
| P50 | 0.94 ms |
| P95 | 1.16 ms |
| P99 | 1.25 ms |
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
| **Epochs trained** | **39** (Early Stop bei Epoch 34) |

**Relevante Dateien:**
- Training Script: [`scripts/train_model.py`](../../scripts/train_model.py)
- Base Config: [`config/base_config.yaml`](../../config/base_config.yaml)
- Checkpoint: [`lightning_logs/M3_Medium_Baseline/version_0/checkpoints/M3_Medium_Baseline-epoch=34-val_loss=0.0013.ckpt`](../../lightning_logs/M3_Medium_Baseline/version_0/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Bemerkung |
|--------|------|-----------|
| **R²** | **0.9046** | Erklärt 90.5% der Varianz |
| **Accuracy** | **87.84%** | Threshold: ±0.05 |
| **RMSE** | 0.0338 | Root Mean Square Error |
| **MAE** | 0.0255 | Mean Absolute Error |
| **MSE** | 0.00114 | Mean Square Error |
| Test Samples | 220,127 | 10% des Datasets |

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **2.05 ms** |
| Std | 0.20 ms |
| Min | 1.78 ms |
| Max | 2.73 ms |
| P50 | 1.96 ms |
| P95 | 2.40 ms |
| P99 | 2.47 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Evaluation Script: [`scripts/evaluate_model.py`](../../scripts/evaluate_model.py)
- Results JSON: [`results/no_dropout/m3/eval.json`](../../results/no_dropout/m3/eval.json)

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

1. [x] M1 Training (Small Baseline) - ✅ R²=0.860, Acc=82.57%
2. [x] M2 Training (Small + Simple Attention) - ✅ R²=0.850, Acc=81.50%
   - Bestätigt: Attention hilft nicht bei kleinen Modellen
3. [x] M3 Training (Medium Baseline) - ✅ R²=0.905, Acc=87.84%
4. [x] M4 Training (Medium + Simple Attention) - ✅ R²=0.919, Acc=90.25% **BEST**
5. [x] M5 Training (Medium + Additive Attention) - ✅ R²=0.907, Acc=88.34%
6. [x] M6 Training (Medium + Scaled Dot-Product) - ✅ R²=0.907, Acc=88.17%
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
| **Epochs trained** | **45** (Early Stop bei Epoch 40) |

**Relevante Dateien:**
- Training Script: [`scripts/train_model.py`](../../scripts/train_model.py)
- Base Config: [`config/base_config.yaml`](../../config/base_config.yaml)
- Checkpoint: [`lightning_logs/M4_Medium_Simple_Attention/version_2/checkpoints/M4_Medium_Simple_Attention-epoch=40-val_loss=0.0011.ckpt`](../../lightning_logs/M4_Medium_Simple_Attention/version_2/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Vergleich zu M3 (Baseline) |
|--------|------|----------------------------|
| **R²** | **0.9191** | +1.60% |
| **Accuracy** | **90.25%** | +2.74% |
| **RMSE** | 0.0311 | -7.99% |
| **MAE** | 0.0236 | -7.45% |
| **MSE** | 0.00097 | -14.91% |
| Test Samples | 220,127 | - |

> **Bestes Modell:** M4 erreicht die höchste Accuracy (90.25%) und R² (0.919) aller getesteten Modelle.

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **2.08 ms** |
| Std | 0.18 ms |
| Min | 1.80 ms |
| Max | 2.71 ms |
| P50 | 2.01 ms |
| P95 | 2.44 ms |
| P99 | 2.53 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Evaluation Script: [`scripts/evaluate_model.py`](../../scripts/evaluate_model.py)
- Results JSON: [`results/no_dropout/m4/eval.json`](../../results/no_dropout/m4/eval.json)
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
| **Epochs trained** | **37** (Early Stop bei Epoch 32) |

**Relevante Dateien:**
- Training Script: [`scripts/train_model.py`](../../scripts/train_model.py)
- Base Config: [`config/base_config.yaml`](../../config/base_config.yaml)
- Checkpoint: [`lightning_logs/M5_Medium_Additive_Attention/version_0/checkpoints/M5_Medium_Additive_Attention-epoch=32-val_loss=0.0012.ckpt`](../../lightning_logs/M5_Medium_Additive_Attention/version_0/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Vergleich zu M3 (Baseline) |
|--------|------|----------------------------|
| **R²** | **0.9074** | +0.31% |
| **Accuracy** | **88.34%** | +0.57% |
| **RMSE** | 0.0332 | -1.78% |
| **MAE** | 0.0251 | -1.57% |
| **MSE** | 0.00111 | -2.64% |
| Test Samples | 220,127 | - |

> **Hinweis:** Die Additive Attention hat die meisten Parameter (+5.5% vs M3), aber nicht die beste Performance. Simple Attention ist effizienter.

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **2.49 ms** |
| Std | 0.22 ms |
| Min | 2.12 ms |
| Max | 3.21 ms |
| P50 | 2.41 ms |
| P95 | 2.88 ms |
| P99 | 3.02 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Evaluation Script: [`scripts/evaluate_model.py`](../../scripts/evaluate_model.py)
- Results JSON: [`results/no_dropout/m5/eval.json`](../../results/no_dropout/m5/eval.json)
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
| **Epochs trained** | **32** (Early Stop bei Epoch 27) |

**Relevante Dateien:**
- Training Script: [`scripts/train_model.py`](../../scripts/train_model.py)
- Base Config: [`config/base_config.yaml`](../../config/base_config.yaml)
- Checkpoint: [`lightning_logs/M6_Medium_Scaled_DP_Attention/version_0/checkpoints/M6_Medium_Scaled_DP_Attention-epoch=27-val_loss=0.0012.ckpt`](../../lightning_logs/M6_Medium_Scaled_DP_Attention/version_0/checkpoints/)

### Test Set Evaluation

| Metrik | Wert | Vergleich zu M3 (Baseline) |
|--------|------|----------------------------|
| **R²** | **0.9068** | +0.24% |
| **Accuracy** | **88.17%** | +0.38% |
| **RMSE** | 0.0334 | -1.18% |
| **MAE** | 0.0252 | -1.18% |
| **MSE** | 0.00111 | -2.64% |
| Test Samples | 220,127 | - |

> **Hinweis:** Scaled Dot-Product (Transformer-Style) zeigt ähnliche Performance wie Additive Attention, beide leicht über der Baseline.

### CPU Inference Zeit

| Metrik | Wert |
|--------|------|
| **Mean** | **2.12 ms** |
| Std | 0.19 ms |
| Min | 1.86 ms |
| Max | 2.65 ms |
| P50 | 2.05 ms |
| P95 | 2.46 ms |
| P99 | 2.53 ms |
| **Target (<10 ms)** | **✅ PASS** |

**Relevante Dateien:**
- Evaluation Script: [`scripts/evaluate_model.py`](../../scripts/evaluate_model.py)
- Results JSON: [`results/no_dropout/m6/eval.json`](../../results/no_dropout/m6/eval.json)
- Attention Weights: [`results/figures/M6_Medium_Scaled_DP_Attention/M6_Medium_Scaled_DP_Attention_attention_weights.npy`](../../results/figures/M6_Medium_Scaled_DP_Attention/)

### Trainingsmetriken (TensorBoard)

**TensorBoard starten:**
```bash
tensorboard --logdir lightning_logs/M6_Medium_Scaled_DP_Attention
```

**Logs:** [`lightning_logs/M6_Medium_Scaled_DP_Attention/version_1/`](../../lightning_logs/M6_Medium_Scaled_DP_Attention/version_1/)

---

*Aktualisiert am: 2026-01-26*
