# EPS Torque Prediction - Projektdokumentation

## Projektziel

Dieses Projekt untersucht den Einsatz von Deep-Learning-Modellen zur **Prädiktion des Lenk-Drehmoments (Steer Torque)** für die elektronische Servolenkung (EPS) eines Fahrzeugs. Basierend auf Fahrzeugzustandsgrößen wird das benötigte EPS-Moment vorhergesagt.

### Anwendungskontext
- **Domäne:** Fahrerassistenzsysteme / Autonomes Fahren
- **Fahrzeug:** Hyundai Sonata 2020
- **Datenquelle:** Telemetriedaten (vermutlich OpenPilot-kompatibel)

### Forschungsfragen
1. Können neuronale Netze das EPS-Moment aus Fahrzeuggrößen prädizieren?
2. Verbessern Attention-Mechanismen die Vorhersagegüte?
3. Wie verhält sich der Trade-off zwischen Modellkomplexität, Genauigkeit und Trainingszeit?

---

## Schnellübersicht

| Aspekt | Details |
|--------|---------|
| **Input** | 5 Features über 50 Zeitschritte |
| **Output** | 1 Wert: `steerFiltered` (normalisiertes, rate-limitiertes Torque) |
| **Modelle** | LSTM, LSTM+Attention (3 Varianten), CNN |
| **Framework** | PyTorch Lightning |
| **Hyperparameter-Tuning** | Optuna |

---

## Dokumentationsstruktur

| Datei | Inhalt |
|-------|--------|
| [data_pipeline.md](data_pipeline.md) | Datenverarbeitung, Features, Preprocessing |
| [models.md](models.md) | Modellarchitekturen im Detail |
| [training.md](training.md) | Training, Hyperparameter, Optuna |
| [evaluation.md](evaluation.md) | Metriken und Evaluierung |
| [configuration.md](configuration.md) | Setup, Dependencies, Pfade |

---

## Projektstruktur

```
att_project/
├── model/                      # Neuronale Netzwerk-Implementierungen
│   ├── LSTM.py                 # Basis-LSTM
│   ├── LSTM_attention.py       # LSTM + Simple Attention
│   ├── CNN_eval.py             # CNN 1D
│   └── diff_attention/         # Erweiterte Attention-Mechanismen
│       ├── additive_attention.py
│       └── scaled_dot_product.py
├── preprocess/                 # Datenaufbereitung
│   ├── data_preprocessing.py   # CSV → DataFrame
│   └── slice_window.py         # Sliding Window Extraktion
├── optuna/                     # Hyperparameter-Optimierung
├── evaluation/                 # Gespeicherte Testergebnisse
├── lightning_logs/             # Trainings-Logs & Checkpoints
├── plot/                       # Visualisierungsscripts
├── attention_visualization/    # Attention Heatmaps
├── main.py                     # Haupt-Trainingsscript
├── data_module.py              # PyTorch Lightning DataModule
└── docs/                       # Diese Dokumentation
```

---

## Implementierte Modelle

| Modell | Beschreibung | Attention-Typ |
|--------|--------------|---------------|
| **LSTMModel** | Baseline LSTM | Keine |
| **LSTMAttentionModel** | LSTM + Simple Attention | Linear → Softmax |
| **LSTMAttentionModel** (Additive) | LSTM + Bahdanau Attention | W·h_i + U·h_j → tanh → v |
| **LSTMScaleDotAttentionModel** | LSTM + Scaled Dot-Product | Q·K^T / √d_k |
| **CNNModel** | 1D CNN mit Global Average Pooling | Keine |

---

## Schnellstart

```python
# Training starten (Beispiel)
from model.LSTM import LSTMModel
from data_module import TimeSeriesDataModule
import pytorch_lightning as pl

data_module = TimeSeriesDataModule(feature_path, target_path, batch_size=32)
model = LSTMModel(input_size=5, hidden_size=128, num_layers=5, output_size=1)

trainer = pl.Trainer(max_epochs=80, accelerator="gpu")
trainer.fit(model, data_module)
```

---

## Status

Dieses Projekt wurde im Rahmen einer Masterarbeit entwickelt. Die Dokumentation dient als Grundlage für weiterführende Arbeiten.
