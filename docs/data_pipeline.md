# Daten-Pipeline

## Übersicht

Die Daten-Pipeline transformiert Fahrzeug-Telemetriedaten in trainierbare Samples für die neuronalen Netze.

```
CSV-Dateien → DataFrame → Sliding Window → Train/Val/Test Split → DataLoader
```

---

## Datenquelle

- **Fahrzeug:** Hyundai Sonata 2020
- **Anzahl CSV-Dateien:** ~20.999 (je nach Konfiguration 5.000 oder mehr)
- **Format:** Zeitreihen mit Fahrzeugzustandsgrößen

---

## Features (Input)

| Feature | Beschreibung | Einheit |
|---------|--------------|---------|
| `vEgo` | Eigengeschwindigkeit des Fahrzeugs | m/s |
| `aEgo` | Längsbeschleunigung | m/s² |
| `steeringAngleDeg` | Aktueller Lenkwinkel | Grad |
| `roll` | Roll-Winkel (Neigung) | rad |
| `latAccelLocalizer` | Laterale Beschleunigung | m/s² |

**Anzahl Features:** 5

---

## Target (Output)

| Target | Beschreibung | Wertebereich |
|--------|--------------|--------------|
| `steerFiltered` | Normalisiertes, rate-limitiertes Lenk-Drehmoment | [-1, 1] |

Das Target repräsentiert das Moment, das die elektronische Servolenkung (EPS) aufbringen soll.

---

## Preprocessing-Schritte

### Parallele Vorverarbeitung (`preprocess/preprocess_parallel.py`)

Das Haupt-Preprocessing-Script kombiniert CSV-Laden und Sliding Window in einem effizienten, parallelisierten Prozess:

```bash
python preprocess/preprocess_parallel.py
```

**Parameter:**
```python
window_size = 50      # Anzahl Zeitschritte im Input
predict_size = 1      # Anzahl Zeitschritte im Output
step_size = 1         # Schrittweite (überlappende Fenster)
```

**Ablauf:**
1. Paralleles Laden der CSV-Dateien
2. Hinzufügen von `sequence_id` pro CSV
3. Gleitendes Fenster über die Zeitreihe
4. Prüfung auf gleiche `sequence_id` (keine Sprünge zwischen Aufnahmen)
5. Filterung nach Fahrbedingungen
6. Speichern als `.npy` (memory-effizient)

### Datenfilterung

Samples werden nur verwendet wenn:
- `latActive == True` → Laterale Regelung aktiv
- `steeringPressed == False` → Fahrer greift nicht ein

Dies stellt sicher, dass nur Situationen verwendet werden, in denen das Assistenzsystem aktiv die Lenkung kontrolliert.

---

## Datenformat

### Input (X)
```
Shape: (num_samples, window_size, num_features)
       (N, 50, 5)
```

Jedes Sample ist eine Matrix mit 50 Zeitschritten und 5 Features.

### Output (Y)
```
Shape: (num_samples, predict_size)
       (N, 1)
```

Jedes Sample hat einen Zielwert: das vorhergesagte Torque.

---

## Data Module (`model/data_module.py`)

PyTorch Lightning DataModule für standardisiertes Laden:

```python
class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, feature_path, target_path, batch_size=32):
        ...
```

### Datensatz-Split

| Split | Anteil | Verwendung |
|-------|--------|------------|
| Training | 70% | Modelltraining |
| Validation | 20% | Hyperparameter-Tuning, Early Stopping |
| Test | 10% | Finale Evaluation |

**Split-Methode:** `torch.utils.data.random_split` (zufällig)

### DataLoader-Konfiguration

```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,          # Nur für Training
    num_workers=15,        # Paralleles Laden
    pin_memory=True        # GPU-Optimierung
)
```

---

## Dateipfade

Alle Pfade werden zentral in `config/settings.py` verwaltet und sind plattformunabhängig:

```python
from config.settings import get_preprocessed_paths, get_raw_data_path, PREPARED_DATASET_DIR

# Standard-Pfade abrufen
paths = get_preprocessed_paths("HYUNDAI_SONATA_2020", window_size=50)
feature_path = paths["features"]
target_path = paths["targets"]
```

### Verzeichnisstruktur

```
att_project/data/
├── dataset/
│   └── HYUNDAI_SONATA_2020/              # Rohe CSV-Dateien
│       ├── *.csv
│       └── ...
└── prepared_dataset/
    └── HYUNDAI_SONATA_2020/
        ├── {N}csv_with_sequence_id.pkl   # Roh-DataFrame
        ├── 50_1_1_sF/
        │   ├── feature_50_1_1_sF.pkl     # Features (X)
        │   ├── target_50_1_1_sF.pkl      # Targets (Y)
        │   ├── sequence_ids_50_1_1_sF.pkl
        │   └── time_steps_50_1_1_sF.pkl
        └── 15_1_1_s/                     # Alternative Fenstergröße
            └── ...
```

**Namenskonvention:** `{window_size}_{predict_size}_{step_size}_{suffix}`

---

## Dataset-Inspektion

Mit `preprocess/inspect_dataset.py` können vorverarbeitete Daten inspiziert werden:

```bash
python preprocess/inspect_dataset.py                    # Standard
python preprocess/inspect_dataset.py --stats            # Nur Statistiken
python preprocess/inspect_dataset.py --index 42         # Bestimmter Index
```

---

## Wichtige Hinweise

1. **Sequenz-Integrität:** Fenster werden nur aus zusammenhängenden Sequenzen erstellt
2. **Normalisierung:** `steerFiltered` ist bereits normalisiert [-1, 1]
3. **Keine Feature-Normalisierung:** Die Input-Features werden nicht explizit normalisiert
4. **Reproduzierbarkeit:** Seed wird in der Config gesetzt (`training.seed: 42`)
