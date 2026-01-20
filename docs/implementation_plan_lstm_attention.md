# Implementierungsplan: LSTM-Attention für EPS Steering Torque Prediction

## Übersicht

Dieses Dokument beschreibt den Implementierungsplan für das Training von 6 Modellen zur Vorhersage von Lenkmoment (Steering Torque) in elektrischen Servolenkungssystemen (EPS).

### Ziel
Systematischer Vergleich von:
1. **Attention-Mechanismen**: Welcher funktioniert am besten?
2. **Effizienz**: Ist Attention effizienter als Modell-Skalierung?

### Hardware
- GPU: NVIDIA RTX 2060 Super
- RAM: 16 GB
- Training: Über Nacht (einzelne Modelle)

---

## Modellübersicht

| ID | Modell | Hidden | Layers | Attention | ~Parameter | ~Trainingszeit |
|----|--------|--------|--------|-----------|------------|----------------|
| M1 | Small Baseline | 64 | 3 | None | 340K | ~2h |
| M2 | Small + Simple | 64 | 3 | Simple | 350K | ~3h |
| M3 | Small + Additive | 64 | 3 | Additive | 380K | ~3h |
| M4 | Small + Scaled Dot-Product | 64 | 3 | Scaled DP | 350K | ~3h |
| M5 | Medium Baseline | 128 | 5 | None | 600K | ~4h |
| M6 | Large Baseline | 256 | 10 | None | 5.0M | ~28h |

---

## Projektstruktur

```
lstm_attention_eps/
├── config/
│   ├── base_config.yaml          # Gemeinsame Konfiguration
│   ├── model_configs/
│   │   ├── m1_small_baseline.yaml
│   │   ├── m2_small_simple_attn.yaml
│   │   ├── m3_small_additive_attn.yaml
│   │   ├── m4_small_scaled_dp_attn.yaml
│   │   ├── m5_medium_baseline.yaml
│   │   └── m6_large_baseline.yaml
│   └── hyperparameter_search.yaml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # Dataset-Klasse für commaSteeringControl
│   │   ├── preprocessing.py      # Sliding Window, Normalisierung
│   │   └── datamodule.py         # PyTorch Lightning DataModule
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_lstm.py          # Baseline LSTM ohne Attention
│   │   ├── attention/
│   │   │   ├── __init__.py
│   │   │   ├── simple.py         # Simple Linear Attention
│   │   │   ├── additive.py       # Bahdanau-style Additive Attention
│   │   │   └── scaled_dot_product.py  # Transformer-style
│   │   └── lstm_attention.py     # LSTM + Attention kombiniert
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training-Loop mit PyTorch Lightning
│   │   ├── callbacks.py          # Early Stopping, Checkpointing
│   │   └── hyperparameter_tuning.py  # Optuna Integration
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py            # Accuracy, RMSE, TCR, LE
│       └── visualization.py      # Training Curves, Attention Weights
├── scripts/
│   ├── train_model.py            # Einzelnes Modell trainieren
│   ├── run_hyperparameter_search.py  # Optuna für ein Modell
│   ├── evaluate_model.py         # Evaluation auf Test-Set
│   └── compare_results.py        # Vergleichstabellen generieren
├── notebooks/
│   └── results_analysis.ipynb    # Ergebnisanalyse und Plots
├── results/
│   ├── checkpoints/              # Gespeicherte Modelle
│   ├── logs/                     # TensorBoard Logs
│   ├── hyperparameter_studies/   # Optuna Studies
│   └── figures/                  # Generierte Plots
├── requirements.txt
└── README.md
```

---

## Arbeitspakete

### AP1: Projektsetup und Daten-Pipeline
**Geschätzte Zeit: 2-3 Stunden**

#### AP1.1: Projektstruktur erstellen
- [ ] Verzeichnisstruktur anlegen
- [ ] `requirements.txt` erstellen:
  ```
  torch>=2.0.0
  pytorch-lightning>=2.0.0
  optuna>=3.0.0
  pandas>=2.0.0
  numpy>=1.24.0
  scikit-learn>=1.3.0
  matplotlib>=3.7.0
  seaborn>=0.12.0
  pyyaml>=6.0
  tensorboard>=2.14.0
  ```
- [ ] Base Config YAML erstellen

#### AP1.2: Daten-Pipeline implementieren
- [ ] Dataset-Klasse für commaSteeringControl
  - Input Features: `speed`, `angle`, `steer_cmd`, `steer_req_actl`, `angle_rate`
  - Target: `torque_eps` (normalisiert)
- [ ] Sliding Window Preprocessing (window_size=50)
- [ ] Train/Val/Test Split (70/20/10)
- [ ] PyTorch Lightning DataModule

#### AP1.3: Daten validieren
- [ ] Testskript für DataModule
- [ ] Prüfen: Shapes, Normalisierung, keine NaN-Werte

**Akzeptanzkriterien:**
- DataModule lädt Daten korrekt
- Batch-Shape: `(batch_size, 50, 5)` für Input, `(batch_size, 1)` für Target

---

### AP2: Modell-Implementierung
**Geschätzte Zeit: 3-4 Stunden**

#### AP2.1: Baseline LSTM
- [ ] `base_lstm.py` implementieren
  - Konfigurierbar: `hidden_size`, `num_layers`, `dropout`
  - Output: Letzter Hidden State → Linear → Prediction
- [ ] Unit-Test: Forward Pass mit Dummy-Daten

#### AP2.2: Attention-Mechanismen
- [ ] `simple.py` - Simple Linear Attention:
  ```python
  # score_i = W * h_i + b (unabhängig pro Zeitschritt)
  # alpha = softmax(scores)
  # context = sum(alpha_i * h_i)
  ```
- [ ] `additive.py` - Additive (Bahdanau) Attention:
  ```python
  # score_ij = v^T * tanh(W * h_i + U * h_j)
  # Paarweise Interaktion zwischen Zeitschritten
  ```
- [ ] `scaled_dot_product.py` - Scaled Dot-Product Attention:
  ```python
  # score_ij = (h_i · h_j) / sqrt(d)
  # Standard Transformer Attention
  ```

#### AP2.3: LSTM + Attention Kombination
- [ ] `lstm_attention.py` implementieren
  - LSTM → Attention → Linear → Prediction
  - Attention-Typ als Parameter
- [ ] Unit-Tests für alle 4 Modellvarianten

**Akzeptanzkriterien:**
- Alle Modelle produzieren Output mit korrekter Shape
- Parameter-Counts entsprechen Erwartungen (±10%)

---

### AP3: Training-Infrastruktur
**Geschätzte Zeit: 2-3 Stunden**

#### AP3.1: PyTorch Lightning Module
- [ ] `trainer.py` mit LightningModule:
  - `training_step`, `validation_step`, `test_step`
  - Loss: MSE
  - Metriken: Accuracy (threshold=0.05), RMSE
  - Optimizer: Adam

#### AP3.2: Callbacks
- [ ] Early Stopping (patience=5, monitor='val_loss')
- [ ] Model Checkpointing (save best)
- [ ] Learning Rate Logging

#### AP3.3: Hyperparameter-Tuning
- [ ] Optuna Integration in `hyperparameter_tuning.py`
- [ ] Suchraum definieren:
  ```yaml
  learning_rate: [1e-4, 1e-2]  # log-uniform
  dropout: [0.0, 0.5]
  batch_size: [32, 64, 128]
  ```
- [ ] TPE Sampler, 25 Trials pro Modell

#### AP3.4: Training-Skript
- [ ] `train_model.py` CLI:
  ```bash
  python scripts/train_model.py --config config/model_configs/m1_small_baseline.yaml
  ```
- [ ] Logging zu TensorBoard
- [ ] Checkpoint-Speicherung

**Akzeptanzkriterien:**
- Training läuft durch ohne Fehler
- Checkpoints werden gespeichert
- TensorBoard zeigt Metriken

---

### AP4: Modell-Konfigurationen
**Geschätzte Zeit: 1 Stunde**

#### AP4.1: Base Config
```yaml
# config/base_config.yaml
data:
  dataset_path: "/pfad/zu/commaSteeringControl"  # TODO: Lokalen Pfad eintragen
  window_size: 50
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1
  
training:
  max_epochs: 80
  early_stopping_patience: 5
  seed: 42
  device: "gpu"  # Training auf GPU
  
evaluation:
  accuracy_threshold: 0.05
  inference_device: "cpu"  # Inferenzzeit auf CPU messen
```

#### AP4.2: Modell-spezifische Configs
- [ ] M1: Small Baseline
  ```yaml
  model:
    type: "baseline"
    hidden_size: 64
    num_layers: 3
    attention: null
  ```
- [ ] M2: Small + Simple Attention
  ```yaml
  model:
    type: "attention"
    hidden_size: 64
    num_layers: 3
    attention: "simple"
  ```
- [ ] M3: Small + Additive Attention
- [ ] M4: Small + Scaled Dot-Product
- [ ] M5: Medium Baseline (hidden=128, layers=5)
- [ ] M6: Large Baseline (hidden=256, layers=10)

---

### AP5: Training durchführen
**Geschätzte Zeit: 4-5 Nächte**

#### AP5.1: Hyperparameter-Suche (optional, aber empfohlen)
Für jedes Modell separat:
```bash
python scripts/run_hyperparameter_search.py --config config/model_configs/m1_small_baseline.yaml --n_trials 25
```

**Empfohlene Reihenfolge:**
| Nacht | Modell | Geschätzte Zeit |
|-------|--------|-----------------|
| 1 | M1 (Small Baseline) | HP-Search + Training: ~4h |
| 1 | M2 (Small + Simple) | HP-Search + Training: ~5h |
| 2 | M3 (Small + Additive) | HP-Search + Training: ~5h |
| 2 | M4 (Small + Scaled DP) | HP-Search + Training: ~5h |
| 3 | M5 (Medium Baseline) | HP-Search + Training: ~6h |
| 4-5 | M6 (Large Baseline) | HP-Search + Training: ~30h |

#### AP5.2: Finales Training
Nach HP-Suche mit besten Parametern:
```bash
python scripts/train_model.py --config config/model_configs/m2_small_simple_attn.yaml --use_best_params
```

#### AP5.3: Checkpoints sichern
- [ ] Beste Modelle nach `results/checkpoints/` kopieren
- [ ] Hyperparameter-Studien dokumentieren

---

### AP6: Evaluation und Analyse
**Geschätzte Zeit: 2-3 Stunden**

#### AP6.1: Test-Set Evaluation
```bash
python scripts/evaluate_model.py --checkpoint results/checkpoints/m2_best.ckpt
```

Metriken sammeln:
- Accuracy (%)
- RMSE
- Anzahl Parameter
- Trainingszeit (GPU)
- Inferenzzeit auf CPU (ms/sample) – Messung: 1000 Samples, Durchschnitt

**Inferenzzeit-Messung:**
```python
# Nach Training: Modell auf CPU laden
model = model.cpu()
model.eval()

# Warm-up
for _ in range(100):
    _ = model(dummy_input)

# Zeitmessung
import time
times = []
for sample in test_samples[:1000]:
    start = time.perf_counter()
    _ = model(sample)
    times.append(time.perf_counter() - start)

inference_time_ms = np.mean(times) * 1000
```

#### AP6.2: Vergleichstabelle generieren
```bash
python scripts/compare_results.py --output results/comparison_table.csv
```

Erwartete Ausgabe:
| Model | Params | Accuracy | RMSE | Train Time | Inference |
|-------|--------|----------|------|------------|-----------|
| M1 Small Baseline | 340K | ~85% | ~0.00116 | ~2h | ~0.5ms |
| M2 Small+Simple | 350K | ~90% | ~0.00098 | ~3h | ~0.6ms |
| ... | ... | ... | ... | ... | ... |

#### AP6.3: Visualisierungen
- [ ] Training Curves (Accuracy vs. Time) für alle Modelle
- [ ] Parameter-Accuracy Trade-off Plot
- [ ] Attention Weight Heatmaps (für M2, M3, M4)

---

### AP7: Paper-Integration
**Geschätzte Zeit: 1-2 Stunden**

#### AP7.1: LaTeX-Tabellen aktualisieren
- [ ] `tab:attention_comparison` mit neuen Werten
- [ ] `tab:complexity` mit korrekten Parameterzahlen
- [ ] `tab:tcr` mit gemessenen Trainingszeiten

#### AP7.2: Plots exportieren
- [ ] `fig:training_curves` als TikZ oder PDF
- [ ] `fig:complexity_tradeoff` aktualisieren

#### AP7.3: Text anpassen
- [ ] Experimentelle Details aktualisieren
- [ ] Konkrete Zahlen einfügen
- [ ] Schlussfolgerungen prüfen

---

## Qualitätssicherung

### Reproduzierbarkeit
- [ ] Fester Seed für alle Experimente (seed=42)
- [ ] Alle Hyperparameter in Config-Dateien dokumentiert
- [ ] Git-Versionierung für Code
- [ ] Bei Bedarf später: Mehrere Seeds für Signifikanztest

### Validierung
- [ ] Sanity Check: Baseline sollte ~85% erreichen
- [ ] Attention-Modelle sollten Baseline übertreffen
- [ ] Parameter-Counts mit theoretischen Werten vergleichen
- [ ] Inferenzzeit auf CPU: Ziel <10ms für Echtzeit-Fähigkeit

### Dokumentation
- [ ] README.md mit Ausführungsanleitung
- [ ] Kommentare im Code
- [ ] Ergebnisse in `results/` strukturiert ablegen
- [ ] Lokaler Pfad zum commaSteeringControl Dataset in Config

---

## Checkliste vor Paper-Submission

- [ ] Alle 6 Modelle erfolgreich trainiert
- [ ] Test-Set Evaluation abgeschlossen
- [ ] Vergleichstabellen vollständig
- [ ] Plots in Paper-Qualität exportiert
- [ ] Code für Reproduzierbarkeit aufgeräumt
- [ ] Hyperparameter dokumentiert

---

## Kontakt / Notizen

**Entscheidungen:**
- **Inferenzzeit:** Auf CPU messen (nach GPU-Training). Begründung: Embedded-Systeme in Fahrzeugen haben typischerweise keine GPU. CPU-Inferenzzeit zeigt Echtzeit-Fähigkeit (Ziel: <10ms für 100Hz Operation).
- **Seeds:** Erstmal 1 Seed (seed=42). Bei Bedarf später erweitern, falls Reviewer Signifikanztests verlangen oder Unterschiede zwischen Modellen sehr klein sind.
- **Dataset:** commaSteeringControl ist lokal vorhanden ✓

**Abhängigkeiten:**
- GPU-Treiber aktuell (CUDA 11.8+ empfohlen)