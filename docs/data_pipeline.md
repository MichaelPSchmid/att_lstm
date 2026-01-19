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

### 1. CSV zu DataFrame (`preprocess/data_preprocessing.py`)

```python
# Lädt alle CSV-Dateien und fügt sequence_id hinzu
# Output: 5000csv_with_sequence_id.pkl
```

- Jede CSV-Datei erhält eine eindeutige `sequence_id`
- Ermöglicht Nachverfolgung der Datenherkunft

### 2. Sliding Window Extraktion (`preprocess/slice_window.py`)

```python
window_size = 50      # Anzahl Zeitschritte im Input
predict_size = 1      # Anzahl Zeitschritte im Output
step_size = 1         # Schrittweite (überlappende Fenster)
```

**Ablauf:**
1. Gleitendes Fenster über die Zeitreihe
2. Prüfung auf gleiche `sequence_id` (keine Sprünge zwischen Aufnahmen)
3. Filterung nach Fahrbedingungen

### 3. Datenfilterung

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

## Data Module (`data_module.py`)

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

Die aktuellen Pfade im Code sind für ein Linux-System konfiguriert:

```
/home/wudamu/MA_tianze/prepared_dataset/HYUNDAI_SONATA_2020/
├── 5000csv_with_sequence_id.pkl          # Roh-DataFrame
├── 50_1_1_sF/
│   ├── feature_50_1_1_sF.pkl             # Features (X)
│   ├── target_50_1_1_sF.pkl              # Targets (Y)
│   ├── sequence_ids_50_1_1_sF.pkl        # Sequence IDs
│   └── time_steps_50_1_1_sF.pkl          # Zeitstempel
└── 15_1_1_s/                             # Alternative Fenstergröße
    └── ...
```

**Namenskonvention:** `{window_size}_{predict_size}_{step_size}_sF`

---

## Verschiedene Fenstergrößen

Es existieren mehrere Varianten des Sliding-Window-Scripts:

| Script | Window Size | Beschreibung |
|--------|-------------|--------------|
| `slice_window.py` | 50 | Standard |
| `slice_window_20999.py` | variabel | Für größeren Datensatz |
| `slice_window_nF.py` | variabel | Alternative Konfiguration |

---

## Wichtige Hinweise

1. **Sequenz-Integrität:** Fenster werden nur aus zusammenhängenden Sequenzen erstellt
2. **Normalisierung:** `steerFiltered` ist bereits normalisiert [-1, 1]
3. **Keine Feature-Normalisierung:** Die Input-Features werden nicht explizit normalisiert
4. **Reproduzierbarkeit:** Seed wird in `main.py` gesetzt (`pl.seed_everything(3407)`)
