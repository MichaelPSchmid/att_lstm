# EPS Torque Prediction - Projektdokumentation

## Projektziel

Dieses Projekt untersucht den Einsatz von Deep-Learning-Modellen zur **Praediktion des Lenk-Drehmoments (Steer Torque)** fuer die elektronische Servolenkung (EPS) eines Fahrzeugs. Basierend auf Fahrzeugzustandsgroessen wird das benoetigte EPS-Moment vorhergesagt.

### Anwendungskontext
- **Domaene:** Fahrerassistenzsysteme / Autonomes Fahren
- **Fahrzeug:** Hyundai Sonata 2020
- **Datenquelle:** Telemetriedaten (OpenPilot-kompatibel)

### Forschungsfragen
1. Koennen neuronale Netze das EPS-Moment aus Fahrzeuggroessen praedizieren?
2. Verbessern Attention-Mechanismen die Vorhersageguete?
3. Wie verhaelt sich der Trade-off zwischen Modellkomplexitaet, Genauigkeit und Inferenzzeit?

---

## Schnelluebersicht

| Aspekt | Details |
|--------|---------|
| **Input** | 5 Features ueber 50 Zeitschritte (5s @ 10 Hz) |
| **Output** | 1 Wert: `steerFiltered` (normalisiertes, rate-limitiertes Torque) |
| **Modelle** | 2 MLP-Baselines, 2 LSTM-Baselines, 3 LSTM+Attention, 1 CNN (8 Modelle) |
| **Evaluation** | Sequenz-Level Split, Block-Bootstrap, 5 Seeds |
| **Framework** | PyTorch Lightning |
| **Hyperparameter-Tuning** | Optuna |

---

## Dokumentationsstruktur

| Datei | Inhalt |
|-------|--------|
| [data_pipeline.md](data_pipeline.md) | Datenverarbeitung, Features, Preprocessing |
| [models.md](models.md) | Modellarchitekturen im Detail |
| [training.md](training.md) | Training, Hyperparameter, Optuna |
| [evaluation_pipeline.md](evaluation_pipeline.md) | Evaluations-Pipeline (Sequenz-Level, 4 Phasen) |
| [configuration.md](configuration.md) | Setup, Dependencies, Pfade |

### Ergebnisberichte

| Datei | Inhalt |
|-------|--------|
| [reports/sequence_level_evaluation_results_no_dropout.md](reports/sequence_level_evaluation_results_no_dropout.md) | Hauptergebnisse (5 Seeds, Sequenz-Level) |
| [reports/attention_analysis.md](reports/attention_analysis.md) | Attention-Weight-Analyse |

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

| ID | Modell | Typ | Parameter | Accuracy |
|----|--------|-----|-----------|----------|
| M1 | MLP Last | MLP (5->64->64->1) | 4,609 | 68.98% |
| M2 | MLP Flat | MLP (250->128->64->1) | 40,449 | 73.44% |
| M3 | Small Baseline | LSTM (64, 3L) | 84,801 | 79.46% |
| M4 | Small + Simple Attn | LSTM+Attn (64, 3L) | 84,866 | 79.41% |
| M5 | Medium Baseline | LSTM (128, 5L) | 597,633 | 79.63% |
| M6 | Medium + Simple Attn | LSTM+Attn (128, 5L) | 597,762 | 79.60% |
| M7 | Medium + Additive Attn | LSTM+Additive (128, 5L) | 630,529 | 79.73% |
| M8 | Medium + Scaled DP | LSTM+ScaledDP (128, 5L) | 597,633 | 79.26% |

> Accuracy: Sequenz-Level Evaluation, 5 Seeds, ohne Dropout.
> Details: [reports/sequence_level_evaluation_results_no_dropout.md](reports/sequence_level_evaluation_results_no_dropout.md)

---

## Schnellstart

```bash
# Training mit Konfigurationsdatei
python scripts/train_model.py --config config/model_configs/m3_small_baseline.yaml

# Training mit Attention-Modell
python scripts/train_model.py --config config/model_configs/m7_medium_additive_attn.yaml

# Batch-Training (alle Modelle, 5 Seeds)
python scripts/batch_runner.py train --variant no_dropout

# Evaluation (Sequenz-Level)
python scripts/batch_runner.py evaluate --variant no_dropout

# Statistische Vergleiche
python scripts/sequence_level_evaluation.py --n-bootstrap 1000 --n-permutations 10000
```

---

## Zentrale Ergebnisse

1. **Sequential Modeling ist entscheidend:** LSTM >> MLP (Cohen's d = 1.08, +10.5 pp Accuracy)
2. **Attention liefert keinen Mehrwert:** Kein Attention-Mechanismus verbessert die LSTM-Baseline signifikant
3. **Scaled Dot-Product schadet:** M8 ist signifikant schlechter als Baseline (p<0.001)
4. **Modellgroesse spielt kaum eine Rolle:** Small (85K) vs Medium (598K) nur +0.17 pp bei 3x Inferenzzeit
5. **Empfehlung:** M3 (Small Baseline LSTM) — bestes Kosten-Nutzen-Verhaeltnis

> Fruhere Evaluation mit Sample-Level-Split zeigte faelschlicherweise Attention-Vorteile.
> Diese waren ein Artefakt von Data Leakage. Details: [evaluation_pipeline.md](evaluation_pipeline.md)

## Status

Dieses Projekt wurde im Rahmen einer Masterarbeit entwickelt.
