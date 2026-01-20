# Model Evaluation Results

> **Ziel:** Vergleich von LSTM-Architekturen mit verschiedenen Attention-Mechanismen für Steering Torque Prediction
>
> **Stand:** 2026-01-20

---

## Übersicht

| Modell | Typ | Parameter | R² | Accuracy | RMSE | Inference (ms) | Status |
|--------|-----|-----------|-----|----------|------|----------------|--------|
| M1 Small Baseline | LSTM (64, 3) | 84,801 | 0.860 | 82.54% | 0.0408 | 0.93 | ✅ |
| M2 Small + Simple Attn | LSTM + Attention (64, 3) | 84,866 | 0.850 | 81.50% | 0.0423 | 1.02 | ✅ |
| M3 Medium Baseline | LSTM (128, 5) | 597,633 | 0.903 | 87.81% | 0.0340 | 2.14 | ✅ |
| M4 Medium + Simple Attn | LSTM + Attention (128, 5) | - | - | - | - | - | ⬜ |
| M5 Medium + Additive Attn | LSTM + Additive (128, 5) | - | - | - | - | - | ⬜ |
| M6 Medium + Scaled DP | LSTM + Scaled DP (128, 5) | - | - | - | - | - | ⬜ |

> **Plan-Änderung:** Basierend auf Paper-Erkenntnissen (Kapitel 5) wurde der Fokus von kleinen auf mittlere Modelle verschoben.
> Bei hidden=64, layers=3 zeigt Attention keinen Vorteil. Bei hidden=128, layers=5 übertrifft Attention die Baseline.

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
2. [x] M2 Training (Small + Simple Attention) - ✅ R²=0.850, Acc=81.50%
   - Bestätigt: Attention hilft nicht bei kleinen Modellen
3. [x] M3 Training (Medium Baseline) - ✅ R²=0.903, Acc=87.81%
4. [ ] M4 Training (Medium + Simple Attention)
5. [ ] M5 Training (Medium + Additive Attention)
6. [ ] M6 Training (Medium + Scaled Dot-Product)
7. [ ] Vergleichstabellen generieren
8. [ ] Attention-Visualisierung erstellen

---

*Generiert am: 2026-01-20*
