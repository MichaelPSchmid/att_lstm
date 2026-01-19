# Konfiguration & Setup

## Dependencies

### Haupt-Bibliotheken

| Bibliothek | Verwendung |
|------------|------------|
| PyTorch | Deep Learning Framework |
| PyTorch Lightning | Training-Framework |
| Optuna | Hyperparameter-Optimierung |
| Pandas | Datenverarbeitung |
| NumPy | Numerische Operationen |
| Matplotlib | Visualisierung |
| TensorBoard | Logging & Monitoring |
| tqdm | Progress Bars |

### Installation

```bash
pip install torch pytorch-lightning optuna pandas numpy matplotlib tensorboard tqdm
```

Oder mit conda:
```bash
conda install pytorch pytorch-lightning -c pytorch
pip install optuna
```

### GPU-Support

Für CUDA-Unterstützung:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Dateipfade

### Aktuell konfigurierte Pfade (Linux)

Die Pfade im Code sind für ein Linux-System konfiguriert:

```python
# In main.py
feature_path = "/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF/feature_50_1_1_sF.pkl"
target_path = "/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF/target_50_1_1_sF.pkl"
```

### Anpassung für Windows

Für lokale Verwendung müssen die Pfade angepasst werden:

```python
# Beispiel für Windows
feature_path = r"C:\Users\...\prepared_dataset\HYUNDAI_SONATA_2020\50_1_1_sF\feature_50_1_1_sF.pkl"
target_path = r"C:\Users\...\prepared_dataset\HYUNDAI_SONATA_2020\50_1_1_sF\target_50_1_1_sF.pkl"
```

### Pfad-Struktur

```
prepared_dataset/
└── HYUNDAI_SONATA_2020/
    ├── 5000csv_with_sequence_id.pkl     # Rohdaten
    ├── 50_1_1_sF/                        # Window=50, Predict=1, Step=1
    │   ├── feature_50_1_1_sF.pkl
    │   ├── target_50_1_1_sF.pkl
    │   ├── sequence_ids_50_1_1_sF.pkl
    │   └── time_steps_50_1_1_sF.pkl
    └── 15_1_1_s/                         # Window=15 Variante
        └── ...
```

---

## Wichtige Konfigurationsparameter

### In `main.py`

```python
# Reproduzierbarkeit
pl.seed_everything(3407)

# GPU-Optimierung
torch.set_float32_matmul_precision('medium')

# Modell
input_size = 5
hidden_size = 128
num_layers = 5
output_size = 1
lr = 0.000382819

# Training
batch_size = 32
max_epochs = 80
early_stop_patience = 5
```

### In `data_module.py`

```python
# DataLoader
batch_size = 32
num_workers = 15
pin_memory = True

# Split
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
```

### In `preprocess/slice_window.py`

```python
window_size = 50
predict_size = 1
step_size = 1
features = ['vEgo', 'aEgo', 'steeringAngleDeg', 'roll', 'latAccelLocalizer']
target = ['steerFiltered']
```

---

## Verzeichnisstruktur

```
att_project/
├── model/                          # Modell-Implementierungen
│   ├── LSTM.py
│   ├── LSTM_attention.py
│   ├── CNN_eval.py
│   ├── diff_attention/
│   │   ├── additive_attention.py
│   │   └── scaled_dot_product.py
│   └── attention_visualization/
├── preprocess/                     # Datenaufbereitung
│   ├── data_preprocessing.py
│   ├── slice_window.py
│   └── slice_window_*.py
├── optuna/                         # Hyperparameter-Tuning
│   ├── optuna1.py
│   ├── optuna2.py
│   └── optuna3.py
├── evaluation/                     # Ergebnisse
├── lightning_logs/                 # Training Logs
├── plot/                           # Visualisierungen
├── attention_visualization/        # Attention Heatmaps
├── comparison/                     # Modellvergleiche
├── prepared_dataset/               # Vorverarbeitete Daten
├── docs/                           # Dokumentation
├── main.py                         # Haupt-Trainingsscript
├── main_hot_map.py                # Attention Visualisierung
├── data_module.py                  # DataModule
└── CLAUDE.md                       # Projekt-Leitfaden
```

---

## Ausführung

### Preprocessing

```bash
# 1. CSV zu DataFrame (falls Rohdaten vorhanden)
python preprocess/data_preprocessing.py

# 2. Sliding Window Extraktion
python preprocess/slice_window.py
```

### Training

```bash
# Standard-Training
python main.py

# Mit TensorBoard Monitoring
tensorboard --logdir lightning_logs &
python main.py
```

### Hyperparameter-Optimierung

```bash
python optuna/optuna1.py
```

### Evaluation

```bash
# Attention Visualisierung
python main_hot_map.py
```

---

## Hardware-Anforderungen

### Minimum

- CPU: Modernes Multi-Core System
- RAM: 16 GB
- Speicher: 10 GB für Daten und Logs

### Empfohlen

- GPU: NVIDIA mit CUDA-Support (GTX 1080 oder besser)
- RAM: 32 GB
- Speicher: SSD für schnelles Datenladen

### Konfiguration für CPU-only

```python
# In main.py
trainer = pl.Trainer(
    accelerator="cpu",
    devices=1,
    # ...
)
```

---

## Bekannte Probleme

### 1. Pfade nicht gefunden

**Problem:** FileNotFoundError bei Datenpfaden

**Lösung:** Pfade in den Scripts anpassen:
- `main.py`
- `optuna/*.py`
- `preprocess/slice_window.py`

### 2. CUDA out of memory

**Problem:** GPU-Speicher reicht nicht

**Lösungen:**
- `batch_size` reduzieren
- `num_workers` reduzieren
- `hidden_size` reduzieren

### 3. num_workers auf Windows

**Problem:** `num_workers > 0` kann auf Windows Probleme verursachen

**Lösung:** In `data_module.py` `num_workers=0` setzen oder multiprocessing-Guard verwenden:
```python
if __name__ == '__main__':
    # Training Code hier
```

---

## Umgebungsvariablen

Optional für bessere Kontrolle:

```bash
# CUDA Device auswählen
export CUDA_VISIBLE_DEVICES=0

# Deterministische Operationen
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

---

## Logging-Verzeichnisse

| Verzeichnis | Inhalt |
|-------------|--------|
| `lightning_logs/` | TensorBoard Logs, Checkpoints |
| `evaluation/` | Gespeicherte Vorhersagen |
| `attention_visualization/` | Attention Heatmaps |

### Logs aufräumen

```bash
# Alte Logs löschen (Vorsicht!)
rm -rf lightning_logs/*/version_*/
```

---

## Versionierung

### Git-ignorierte Dateien

Typisch zu ignorieren (in `.gitignore`):
```
lightning_logs/
*.pkl
__pycache__/
*.pyc
.ipynb_checkpoints/
```

### Checkpoint-Management

Checkpoints können groß werden. Empfehlung:
- Nur beste Modelle behalten (`save_top_k=3`)
- Alte Versionen regelmäßig löschen
