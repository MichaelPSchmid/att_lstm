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

## Zentrale Pfad-Konfiguration

Alle Pfade werden zentral in `config/settings.py` verwaltet. Das Projekt ist plattformunabhängig (Windows/Linux/macOS).

### config/settings.py

```python
from config.settings import (
    PROJECT_ROOT,           # Projektverzeichnis
    DATA_ROOT,              # data/
    DATASET_DIR,            # data/dataset/
    PREPARED_DATASET_DIR,   # data/prepared_dataset/
    LIGHTNING_LOGS_DIR,     # lightning_logs/
    FEATURE_PATH,           # Standard-Feature-Pfad
    TARGET_PATH,            # Standard-Target-Pfad
)

# Für spezifische Konfigurationen:
from config.settings import get_preprocessed_paths, get_raw_data_path
```

### Pfade konfigurieren

Die Daten liegen standardmäßig im `data/` Verzeichnis innerhalb des Projekts:

```
att_project/
└── data/
    ├── dataset/                    # Rohe CSV-Dateien
    │   └── HYUNDAI_SONATA_2020/
    └── prepared_dataset/           # Vorverarbeitete Pickle-Dateien
        └── HYUNDAI_SONATA_2020/
            └── 50_1_1_sF/
```

### Verzeichnisse erstellen

```bash
python -c "from config.settings import ensure_dirs_exist; ensure_dirs_exist()"
```

Oder:
```python
from config.settings import ensure_dirs_exist
ensure_dirs_exist()
```

### Konfiguration anzeigen

```bash
python -c "from config.settings import print_config; print_config()"
```

Output:
```
============================================================
Project Configuration
============================================================
PROJECT_ROOT:         C:\Users\...\att_project
DATA_ROOT:            C:\Users\...\att_project\data
FEATURE_PATH:         C:\Users\...\att_project\data\prepared_dataset\...
============================================================
```

---

## Verzeichnisstruktur

```
att_project/
├── config/                         # Konfiguration
│   ├── settings.py                 # Zentrale Pfad-Konfiguration
│   ├── loader.py                   # Config-Loader für YAML
│   ├── base_config.yaml            # Basis-Konfiguration
│   └── model_configs/              # Modell-spezifische Configs
├── data/                           # Datenverzeichnis
│   ├── dataset/                    # Rohe CSV-Dateien
│   │   └── HYUNDAI_SONATA_2020/
│   └── prepared_dataset/           # Vorverarbeitete Daten
│       └── HYUNDAI_SONATA_2020/
│           ├── {N}csv_with_sequence_id.pkl
│           └── 50_1_1_sF/
├── model/                          # Modell-Implementierungen
│   ├── lstm_baseline.py
│   ├── lstm_simple_attention.py
│   ├── lstm_additive_attention.py
│   ├── lstm_scaled_dp_attention.py
│   └── data_module.py
├── scripts/                        # Ausführbare Skripte
│   ├── train_model.py
│   └── evaluate_model.py
├── preprocess/                     # Datenaufbereitung
│   ├── preprocess_parallel.py
│   └── inspect_dataset.py
├── optuna/                         # Hyperparameter-Tuning
├── lightning_logs/                 # Training Logs & Checkpoints
├── plot/                           # Visualisierungen
└── docs/                           # Dokumentation
```

---

## Wichtige Konfigurationsparameter

### In `config/base_config.yaml`

```yaml
training:
  seed: 42
  max_epochs: 80
  batch_size: 32
  learning_rate: 0.0001
  early_stopping:
    patience: 5
    monitor: val_loss

data:
  vehicle: HYUNDAI_SONATA_2020
  window_size: 50
  predict_size: 1
  step_size: 1
  variant: full
```

### In `model/data_module.py`

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

### In `preprocess/preprocess_parallel.py`

```python
window_size = 50
predict_size = 1
step_size = 1
features = ['vEgo', 'aEgo', 'steeringAngleDeg', 'roll', 'latAccelLocalizer']
target = ['steerFiltered']
```

---

## Ausführung

### Ersteinrichtung

```bash
# 1. Verzeichnisse erstellen
python -c "from config.settings import ensure_dirs_exist; ensure_dirs_exist()"

# 2. Rohdaten in data/dataset/HYUNDAI_SONATA_2020/ kopieren

# 3. Preprocessing
python preprocess/preprocess_parallel.py
```

### Training

```bash
# Training mit Konfigurationsdatei
python scripts/train_model.py --config config/model_configs/m1_small_baseline.yaml

# Mit TensorBoard Monitoring
tensorboard --logdir lightning_logs &
python scripts/train_model.py --config config/model_configs/m2_small_simple_attn.yaml
```

### Hyperparameter-Optimierung

```bash
python optuna/optuna1.py
```

### Evaluation

```bash
python scripts/evaluate_model.py --checkpoint path/to/checkpoint.ckpt --config config/model_configs/m1_small_baseline.yaml
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

### 1. Daten nicht gefunden

**Problem:** FileNotFoundError bei Datenpfaden

**Lösung:**
1. Verzeichnisse erstellen: `python -c "from config.settings import ensure_dirs_exist; ensure_dirs_exist()"`
2. Daten in `data/dataset/` kopieren
3. Preprocessing ausführen: `python preprocess/preprocess_parallel.py`

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
# CUDA Device auswählen (Linux/macOS)
export CUDA_VISIBLE_DEVICES=0

# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES=0

# Deterministische Operationen
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

---

## Logging-Verzeichnisse

| Verzeichnis | Inhalt |
|-------------|--------|
| `lightning_logs/` | TensorBoard Logs, Checkpoints |
| `results/` | Evaluationsergebnisse, Plots |

### Logs aufräumen

```bash
# Linux/macOS
rm -rf lightning_logs/*/version_*/

# Windows PowerShell
Remove-Item -Recurse lightning_logs\*\version_*
```

---

## Versionierung

### Git-ignorierte Dateien

In `.gitignore`:
```
# Daten (zu groß für Git)
data/dataset/
data/prepared_dataset/
*.pkl

# Logs und Checkpoints
lightning_logs/

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
```

### Checkpoint-Management

Checkpoints können groß werden. Empfehlung:
- Nur beste Modelle behalten (`save_top_k=3`)
- Alte Versionen regelmäßig löschen
