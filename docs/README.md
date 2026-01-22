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
├── config/                     # Konfiguration
│   ├── settings.py             # Zentrale Pfad-Konfiguration
│   ├── loader.py               # Config-Loader für YAML
│   ├── base_config.yaml        # Basis-Konfiguration
│   └── model_configs/          # Modell-spezifische Configs
├── model/                      # Neuronale Netzwerk-Implementierungen
│   ├── lstm_baseline.py        # Basis-LSTM
│   ├── lstm_simple_attention.py    # LSTM + Simple Attention
│   ├── lstm_additive_attention.py  # LSTM + Additive Attention
│   ├── lstm_scaled_dp_attention.py # LSTM + Scaled Dot-Product
│   └── data_module.py          # PyTorch Lightning DataModule
├── scripts/                    # Ausführbare Skripte
│   ├── train_model.py          # Haupt-Trainingsscript
│   └── evaluate_model.py       # Evaluationsscript
├── preprocess/                 # Datenaufbereitung
│   ├── preprocess_parallel.py  # Parallele Vorverarbeitung
│   └── inspect_dataset.py      # Dataset-Inspektion
├── optuna/                     # Hyperparameter-Optimierung
├── lightning_logs/             # Trainings-Logs & Checkpoints
├── plot/                       # Visualisierungsscripts
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

```bash
# Training mit Konfigurationsdatei
python scripts/train_model.py --config config/model_configs/m1_small_baseline.yaml

# Training mit Attention-Modell
python scripts/train_model.py --config config/model_configs/m2_small_simple_attn.yaml

# Evaluation
python scripts/evaluate_model.py --checkpoint path/to/checkpoint.ckpt --config config/model_configs/m1_small_baseline.yaml
```

Oder programmatisch:
```python
from model.lstm_baseline import LSTMModel
from model.data_module import TimeSeriesDataModule
import pytorch_lightning as pl

data_module = TimeSeriesDataModule(feature_path, target_path, batch_size=32)
model = LSTMModel(input_size=5, hidden_size=128, num_layers=5, output_size=1)

trainer = pl.Trainer(max_epochs=80, accelerator="gpu")
trainer.fit(model, data_module)
```

---

## Status

Dieses Projekt wurde im Rahmen einer Masterarbeit entwickelt. Die Dokumentation dient als Grundlage für weiterführende Arbeiten.
