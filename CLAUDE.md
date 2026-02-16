# ATT_PROJECT - EPS Torque Prediction mit Attention-Vergleich

> **Dokumentation:** `docs/README.md`
> **Aktuelle Notizen:** `docs/aktuell/`

## Session Start

**Bevor du startest:**
1. Gibt es einen Notizzettel in `docs/aktuell/`? Lesen!
2. Neues Feature? Notizzettel anlegen (siehe [Task-Tracking](#task-tracking))
3. Check `docs/README.md` fuer Projektuebersicht

---

## Kernregel: Verstehen vor Handeln

**Bevor du Code schreibst, aenderst oder hinzufuegst:**
1. **Lies** den existierenden Code
2. **Verstehe** warum er so ist
3. **Aendere** an der richtigen Stelle

### Warnsignale

Wenn du dabei bist, eines davon zu tun, **STOPP**:

- "Ich erstelle eine neue Version von..."
- "Ich wrappe das mal um..."
- "Ich bin nicht sicher was das tut, aber..."
- "Das cast ich einfach..."
- "Ich fuege das hier schnell ein..."

Diese Impulse zeigen: Du hast noch nicht genug verstanden.

### Beispiele

```python
# FALSCH: Wrapper statt Fix
def train_model_v2(config):
    ...

# RICHTIG: Original fixen
def train_model(config):
    ...

# FALSCH: Unsafe Type-Ignorierung
result = tensor  # type: ignore

# RICHTIG: Proper Type Handling
if not isinstance(tensor, torch.Tensor):
    raise TypeError(f"Expected Tensor, got {type(tensor)}")
result = tensor

# FALSCH: Pfade hardcoden
features = pd.read_pickle("data/prepared_dataset/HYUNDAI_SONATA_2020/50_1_1_sF/features.pkl")

# RICHTIG: Durch config/settings.py
from config.settings import get_preprocessed_paths
paths = get_preprocessed_paths(variant="paper")
features = pd.read_pickle(paths["features"])
```

### Vor dem Erstellen neuer Dateien

**Pruefe zuerst:**
1. Gibt es bereits eine Datei fuer diesen Zweck?
2. Kann ich eine existierende Datei erweitern?
3. Folgt mein Dateiname dem bestehenden Pattern?

**Typische Fehler:**
- `lstm_baseline_v2.py` statt `lstm_baseline.py` zu fixen
- `scripts/new_evaluate.py` statt `scripts/evaluate_model.py` zu erweitern
- `scripts/helpers.py` statt in `scripts/shared/` einzufuegen

**Grep vor Create:**
```bash
# Bevor du erstellst, suche nach Aehnlichem
grep -r "class.*Model" model/
grep -r "def evaluate" scripts/
```

---

## Projektziel

Praediktion des EPS-Lenk-Drehmoments (Steer Torque) fuer die elektronische Servolenkung eines Hyundai Sonata 2020. Vergleich von LSTM-Baselines mit verschiedenen Attention-Mechanismen.

### Forschungsfragen
1. Koennen neuronale Netze das EPS-Moment aus Fahrzeuggroessen praedizieren?
2. Verbessern Attention-Mechanismen die Vorhersageguete?
3. Trade-off zwischen Modellkomplexitaet, Genauigkeit und Inferenzzeit?

### Architektur

```
CSV-Fahrsequenzen -> Preprocessing -> Sliding Window (50 Steps, 5 Features)
                                          |
                            Train/Val/Test Split (Sequenz-Level)
                                          |
                          8 Modelle (M1-M8): MLP, LSTM, LSTM+Attention
                                          |
                          Evaluation: Sequenz-Level, Block-Bootstrap, 5 Seeds
```

### Kernparameter

| Parameter | Wert |
|-----------|------|
| Input | 5 Features, 50 Zeitschritte (5s @ 10 Hz) |
| Output | 1 Wert: `steerFiltered` (normalisiertes Torque, [-1, 1]) |
| Features | vEgo, aEgo, steeringAngleDeg, roll, latAccelLocalizer |
| Split | 70/20/10 (Train/Val/Test), Sequenz-Level, split_seed=0 |
| Seeds | [42, 94, 123, 7, 231] |
| Accuracy-Schwelle | 0.05 (5% des normalisierten Bereichs) |
| Framework | PyTorch Lightning |
| Batch Size | 32 |
| Max Epochs | 80 |
| Early Stopping | Patience 15 |

### Modelle (8)

| ID | Name | Architektur | Parameter |
|----|------|-------------|-----------|
| M1 | MLP Last | MLP (5->64->64->1) | 4.6K |
| M2 | MLP Flat | MLP (250->128->64->1) | 40K |
| M3 | Small Baseline | LSTM (64, 3 Layers) | 85K |
| M4 | Small + Simple Attn | LSTM + Simple Attention (64, 3L) | 85K |
| M5 | Medium Baseline | LSTM (128, 5 Layers) | 598K |
| M6 | Medium + Simple Attn | LSTM + Simple Attention (128, 5L) | 598K |
| M7 | Medium + Additive Attn | LSTM + Bahdanau Attention (128, 5L) | 631K |
| M8 | Medium + Scaled DP | LSTM + Scaled Dot-Product (128, 5L) | 598K |

> Modell-Details: `docs/models.md`
> Konfigurationen: `config/model_configs/`

---

## Arbeitsanweisungen

### Code-Stil

```python
# Docstrings: Google-Style (ENGLISCH)
def function(param: type) -> return_type:
    """Short description.

    Args:
        param: Description

    Returns:
        Description

    Raises:
        ValueError: If param is invalid
    """

# Type Hints: IMMER verwenden
from typing import List, Dict, Optional, Tuple

# Logging: logging Modul, KEIN print()
import logging
logger = logging.getLogger(__name__)
logger.info("Training model %s with seed %d", model_name, seed)

# Error Handling: Spezifische Exceptions
class ConfigError(ValueError):
    """Raised when configuration is invalid."""
    pass
```

### Verzeichnisstruktur

```
att_project/
|-- config/                     # Konfiguration
|   |-- settings.py             # Zentrale Pfad-Konfiguration
|   |-- loader.py               # Config-Loader fuer YAML
|   |-- base_config.yaml        # Basis-Konfiguration
|   +-- model_configs/          # Modell-spezifische Configs (m1-m8)
|-- model/                      # Neuronale Netzwerk-Implementierungen
|   |-- lstm_baseline.py        # LSTM Baseline (M3, M5)
|   |-- lstm_simple_attention.py    # LSTM + Simple Attention (M4, M6)
|   |-- lstm_additive_attention.py  # LSTM + Additive/Bahdanau (M7)
|   |-- lstm_scaled_dp_attention.py # LSTM + Scaled Dot-Product (M8)
|   |-- mlp_baseline.py         # MLP Baselines (M1, M2)
|   +-- data_module.py          # PyTorch Lightning DataModule
|-- scripts/                    # Ausfuehrbare Skripte
|   |-- train_model.py          # Training einzelner Modelle
|   |-- batch_runner.py         # Batch-Training/Evaluation (alle Modelle)
|   |-- evaluate_model.py       # Modell-Evaluation
|   |-- sequence_level_evaluation.py  # Statistische Vergleiche
|   |-- bootstrap_evaluation.py # Bootstrap CIs & Permutationstests
|   |-- compare_results.py      # Ergebnis-Vergleiche
|   |-- generate_figures.py     # Visualisierungen
|   |-- threshold_sensitivity.py # Schwellwert-Analyse
|   |-- shared/                 # Geteilte Utilities
|   |   |-- models.py           # Modell-Registry (Single Source of Truth)
|   |   |-- metrics.py          # Metriken-Berechnung
|   |   |-- checkpoints.py      # Checkpoint-Verwaltung
|   |   +-- paths.py            # Pfad-Utilities
|   +-- callbacks/
|       +-- attention_callback.py  # Attention-Weights speichern
|-- preprocess/                 # Datenaufbereitung
|   |-- preprocess_parallel.py  # Parallele Vorverarbeitung
|   +-- inspect_dataset.py      # Dataset-Inspektion
|-- tests/                      # Pytest Tests
|-- notebooks/                  # Jupyter Notebooks
|-- data/                       # Daten (nicht in Git)
|   |-- dataset/                # Rohe CSV-Dateien
|   +-- prepared_dataset/       # Vorverarbeitete Pickle-Dateien
|-- results/                    # Experiment-Ergebnisse
|   |-- no_dropout/             # Ergebnisse ohne Dropout
|   |-- bootstrap/              # Bootstrap/Statistik-Ergebnisse
|   +-- paper/                  # Paper-Figures
|-- docs/                       # Dokumentation
|   |-- aktuell/                # Session-Notizen (temporaer)
|   |-- archiv/                 # Abgeschlossene Notizen
|   |-- reports/                # Ergebnisberichte
|   +-- paper/                  # Paper-Referenzen
|-- optuna/                     # Hyperparameter-Optimierung
|-- lightning_logs/             # Training Logs & Checkpoints
|-- plot/                       # Visualisierungsscripts
+-- attention_weights/          # Gespeicherte Attention-Weights
```

### Test-First Workflow

1. **Schreibe erst den Test**
   ```bash
   pytest tests/test_new_feature.py -v
   ```

2. **Implementiere die Funktion**
   ```python
   # Code in model/ oder scripts/
   # Mit Type Hints, Docstrings, Logging
   ```

3. **Pruefe ob Tests gruen sind**
   ```bash
   pytest tests/ -v
   ```

4. **Dokumentiere (VOR Commit)**
   - Notizzettel in `docs/aktuell/` aktualisieren
   - NICHT: CLAUDE.md aendern (ausser bei Methodik-Aenderung)

5. **Commit mit aussagekraeftiger Message**
   ```bash
   git commit -m "feat(eval): Sequenz-Level Bootstrap implementiert"
   ```

---

## Task-Tracking

### Zwei-Ebenen-System

| Datei | Zweck | Lebensdauer | Wer pflegt? |
|-------|-------|-------------|-------------|
| **CLAUDE.md** | Arbeitsanweisungen, Konventionen (statisch) | Permanent | Bei Methodik-Aenderung |
| **docs/aktuell/*.md** | Session-Kontext, TODOs (temporaer) | Bis Feature fertig | Waehrend Arbeit |

### Wann Notizzettel anlegen?

**Immer wenn:**
- Neues groesseres Feature startest
- Nach Pause/Unterbrechung zurueckkommst
- Aufgabe umfangreicher ist (>1h Arbeit)
- Mehrere Teilaufgaben parallel laufen

**Template fuer Notizzettel:**

```markdown
# [Titel] - [Datum]

## Status
- Aktuelle Phase: ...
- Naechster Schritt: ...

## Kontext
- Warum arbeite ich daran?
- Was ist die Herausforderung?

## Offene Aufgaben
- [ ] Schritt 1
- [ ] Schritt 2

## Entscheidungen/Erkenntnisse
- [Timestamp] Entscheidung: ...
- [Timestamp] Problem: ... -> Loesung: ...

## Naechste Session
- Wo weitermachen?
- Was fehlt noch?
```

**Workflow:**
```bash
# 1. Notizzettel erstellen
# Dateiname: thema-YYYY-MM-DD.md

# 2. Waehrend Arbeit aktualisieren
# - Checkboxen abhaken
# - Erkenntnisse notieren

# 3. Bei Abschluss archivieren
mv docs/aktuell/thema-2026-02-16.md docs/archiv/
```

---

## Pruef-Checkliste

**Bevor du eine Aufgabe als erledigt meldest:**

### 1. Funktioniert es?

```bash
# Tests
pytest tests/ -v

# Spezifischer Test
pytest tests/test_data_module.py::TestSequenceLevelSplit -v

# Code-Qualitaet
black model/ scripts/ tests/
ruff check model/ scripts/
```

### 2. Ist es vollstaendig?

| Pruefen | Bei Problem |
|---------|-------------|
| Alle betroffenen Stellen gefunden? | `grep -r "function_name" model/ scripts/` |
| Imports aktualisiert? | `__init__.py` Dateien pruefen |
| Kein toter Code? | Ungenutzte Imports/Funktionen entfernen |
| Config angepasst? | `config/base_config.yaml` pruefen |

### 3. Ist es sauber?

| Pruefen | Bei Problem |
|---------|-------------|
| Folgt bestehenden Patterns? | Vergleiche mit aehnlichem Code |
| Type Hints vollstaendig? | Alle public Functions/Classes |
| Docstrings vorhanden? | Alle public Functions/Classes |
| Logging statt print()? | `grep -r "print(" model/ scripts/` |

### 4. Bei Problemen

```
Tests schlagen fehl?
|-- pytest -v fuer Details
|-- pytest --pdb fuer Debugging
+-- Einzelnen Test isoliert laufen lassen

In einer Sackgasse?
|-- STOPP - nicht weiterwursteln
|-- Notizzettel: Problem dokumentieren
|-- Gibt es einfacheren Weg?
+-- Bei Unsicherheit: Fragen

Mehrere Ansaetze moeglich?
|-- Gibt es bestehendes Pattern im Code?
|-- Wenn ja -> diesem folgen
|-- Wenn nein -> einfachsten Ansatz waehlen
+-- Bei Architektur-Entscheidung: Fragen
```

---

## Dokumentations-Pflicht

**Nach abgeschlossener Aufgabe:**

### 1. Notizzettel aktualisieren

```markdown
## Status
- Phase 1 (DataModule Fix): KOMPLETT
- Naechster: Phase 2 (Training)
```

### 2. Git Commit (nach Bestaetigung)

```bash
# Format: Conventional Commits
git commit -m "feat(eval): Sequenz-Level Evaluation implementiert"
git commit -m "fix(data): Split-Seed Parameter weitergeben"
git commit -m "test(eval): Bootstrap-Tests hinzugefuegt"
git commit -m "docs: Evaluation-Pipeline Dokumentation"
git commit -m "refactor(model): Attention-Module vereinheitlicht"
```

### 3. Wann CLAUDE.md aendern?

**NUR bei:**
- Neue Arbeitsweise/Methodik
- Neue Konventionen
- Neue Tools/Dependencies
- Strukturelle Aenderungen am Projekt

**NICHT bei:**
- Status-Updates (Notizzettel)
- TODOs (Notizzettel)
- Code-Aenderungen

---

## Claude Code Workflows

### Typische Kommandos

```bash
# Training
python scripts/train_model.py --config config/model_configs/m3_small_baseline.yaml
python scripts/train_model.py --config config/model_configs/m7_medium_additive_attn.yaml --save-attention

# Batch-Training (alle Modelle, 5 Seeds)
python scripts/batch_runner.py train --variant no_dropout
python scripts/batch_runner.py train --variant no_dropout --models m3 m5 m7

# Evaluation
python scripts/batch_runner.py evaluate --variant no_dropout
python scripts/evaluate_model.py --checkpoint path/to/checkpoint.ckpt --config config/model_configs/m3_small_baseline.yaml

# Statistische Vergleiche (Sequenz-Level)
python scripts/sequence_level_evaluation.py --n-bootstrap 1000 --n-permutations 10000

# Batch: Train + Evaluate
python scripts/batch_runner.py all --variant no_dropout

# Verfuegbare Modelle/Checkpoints auflisten
python scripts/batch_runner.py list --variant no_dropout

# Tests
pytest tests/ -v
pytest tests/test_data_module.py -v
pytest tests/test_sequence_level_evaluation.py -v

# Code-Qualitaet
black model/ scripts/ tests/
ruff check model/ scripts/

# Projekt-Status
git status
git diff
git log --oneline -10
```

### Bei Fehlern

```bash
# Test fehlgeschlagen
pytest tests/test_X.py::test_func -v   # Einzelnen Test
pytest --pdb                            # Debugger bei Fehler

# Import-Fehler
python -c "from config.settings import print_config; print_config()"
pip list | grep torch

# CUDA-Probleme
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Konventionen

- **Sprache Code:** Englisch (Docstrings, Variablen, Funktionen)
- **Sprache Docs:** Deutsch (ausser Docstrings im Code)
- **Commit-Messages:** Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`)
- **Branch-Namen:** `feature/beschreibung`, `fix/beschreibung`
- **Test-Dateien:** `test_<module>.py` in `tests/`
- **Config-Dateien:** YAML in `config/model_configs/`
- **Modell-Registry:** `scripts/shared/models.py` ist Single Source of Truth fuer Modell-IDs
- **Pfade:** Immer ueber `config/settings.py`, nie hardcoden

---

## Wichtige Architektur-Entscheidungen

- **Sequenz-Level Split:** Train/Val/Test Split auf Sequenz-Ebene (nicht Sample-Ebene), um Data Leakage zu vermeiden. `split_seed=0` ist unabhaengig vom Trainings-Seed.
- **5 Seeds:** Jedes Modell wird mit 5 verschiedenen Seeds trainiert fuer robuste Statistik.
- **Sequenz-Level Evaluation:** Metriken werden pro Testsequenz aggregiert (~500 unabhaengige Datenpunkte), dann Block-Bootstrap und Permutationstests auf dieser Ebene.
- **Keine Dropout-Variante als Default:** Die Hauptergebnisse verwenden `no_dropout`. Dropout-Varianten existieren als Vergleich.

---

## Bei Problemen

1. **Pruefe Notizzettel** in `docs/aktuell/` fuer aktuellen Kontext
2. **Pruefe `docs/README.md`** fuer Projektuebersicht
3. **Bei Unklarheiten: Frage nach, bevor du implementierst**
4. **Modell-Definitionen:** `scripts/shared/models.py`
5. **Pfad-Konfiguration:** `config/settings.py`

---

## Schnellreferenz

| Was? | Wo? |
|------|-----|
| Arbeitsanweisungen | Diese Datei |
| Projektuebersicht | `docs/README.md` |
| Modellarchitekturen | `docs/models.md` |
| Daten-Pipeline | `docs/data_pipeline.md` |
| Training-Doku | `docs/training.md` |
| Evaluation-Pipeline | `docs/evaluation_pipeline.md` |
| Setup & Konfiguration | `docs/configuration.md` |
| Session-Kontext | `docs/aktuell/*.md` |
| Ergebnisberichte | `docs/reports/` |
| Modell-Registry | `scripts/shared/models.py` |
| Basis-Konfiguration | `config/base_config.yaml` |
| Modell-Configs | `config/model_configs/` |
| Pfad-Konfiguration | `config/settings.py` |
| Tests | `tests/` |

---

*Letzte Aktualisierung: 2026-02-16*
