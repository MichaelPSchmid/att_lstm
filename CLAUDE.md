# WEND - Wavelet-based Event Novelty Detection

> **Detaillierter Plan:** `docs/IMPLEMENTIERUNGSPLAN.md`  
> **Fortschritt:** `docs/PROGRESS.md`  
> **Aktuelle Notizen:** `docs/aktuell/`

## Session Start

**Bevor du startest:**
1. Gibt es einen Notizzettel in `docs/aktuell/`? → Lesen!
2. Neues Feature/AP? → Notizzettel anlegen (siehe [Task-Tracking](#task-tracking))
3. Check `docs/PROGRESS.md` für letzten abgeschlossenen Checkpoint

---

## Kernregel: Verstehen vor Handeln

**Bevor du Code schreibst, änderst oder hinzufügst:**
1. **Lies** den existierenden Code
2. **Verstehe** warum er so ist
3. **Ändere** an der richtigen Stelle

### Warnsignale

Wenn du dabei bist, eines davon zu tun → **STOPP**:

- "Ich erstelle eine neue Version von..."
- "Ich wrappe das mal um..."
- "Ich bin nicht sicher was das tut, aber..."
- "Das cast ich einfach..."
- "Ich füge das hier schnell ein..."

Diese Impulse zeigen: Du hast noch nicht genug verstanden.

### Beispiele

```python
# ❌ Wrapper statt Fix
def generate_event_v2(params):
    ...

# ✅ Original fixen
def generate_event(params):
    ...

# ❌ Unsafe Type-Ignorierung
result = signal  # type: ignore

# ✅ Proper Type Handling
if not isinstance(signal, np.ndarray):
    raise TypeError(f"Expected ndarray, got {type(signal)}")
result = signal

# ❌ Direkter Zugriff überall
events = generator.generate_events(...)

# ✅ Durch bestehende Struktur
dataset = EventDataset.from_config(config)
events = dataset.train_events
```

### Vor dem Erstellen neuer Dateien

**Prüfe zuerst:**
1. Gibt es bereits eine Datei für diesen Zweck?
2. Kann ich eine existierende Datei erweitern?
3. Folgt mein Dateiname dem bestehenden Pattern?

**Typische Fehler:**
- `generator_v2.py` statt `generator.py` zu fixen
- `utils/new_helpers.py` statt in passende `utils/*.py` einzufügen
- `features/swt_extract.py` statt `features/extractor.py` zu erweitern

**Grep vor Create:**
```bash
# Bevor du erstellst, suche nach Ähnlichem
grep -r "class.*Generator" wend/
grep -r "def extract_features" wend/
```

---

## Projektziel

Pipeline für unüberwachte Novelty Detection in hochfrequenten Zeitseriendaten (2000 Hz) mit Dataset-Completeness-Schätzung.

**Aktueller Status:** Phase 1 - Synthetischer Proof of Concept  
**Check:** `docs/PROGRESS.md` für Details

### Architektur

```
Signal → Event-Extraktion → SWT → Features (74-dim) → k-NN Novelty → Coverage
                              ↓
                    Duration-Klassen (5)
```

### Kernparameter

| Parameter | Wert | 
|-----------|------|
| k (k-NN) | 5 |
| Wavelet | coif5, Level 5 |
| Duration-Klassen | 5 (IMPULSE, TRANSIENT, SHORT, MEDIUM, LONG) |
| Event-Typen | 15 (3 pro Klasse) |
| Power-law α | 1.5 |
| SNR Default | 20 dB |
| Train/Test | 70/30 stratified |

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
import numpy.typing as npt

# Dataclasses: Für Datenstrukturen
from dataclasses import dataclass

@dataclass
class Event:
    event_id: int
    signal: npt.NDArray[np.float64]
    ...

# Logging: logging Modul, KEIN print()
import logging
logger = logging.getLogger(__name__)
logger.info("Processing event %d", event_id)

# Error Handling: Spezifische Exceptions
class InvalidEventError(ValueError):
    """Raised when event validation fails."""
    pass
```

### Verzeichnisstruktur

```
wend/
├── wend/                    # Hauptpaket
│   ├── data/                # AP1: Event-Generator
│   ├── features/            # AP2: Feature-Extraktion  
│   ├── detection/           # AP3: Novelty Detection
│   ├── coverage/            # AP4: Coverage Estimation
│   ├── evaluation/          # AP5: Evaluation
│   └── utils/               # Hilfsfunktionen
├── tests/                   # Pytest Tests
├── notebooks/               # Jupyter Notebooks
├── configs/                 # YAML Konfigurationen
├── docs/                    # Dokumentation
│   ├── aktuell/             # Session-Notizen (temporär)
│   ├── archiv/              # Abgeschlossene Notizen
│   ├── IMPLEMENTIERUNGSPLAN.md
│   ├── PROGRESS.md          # Checkpoint-Tracking
│   └── ARCHITECTURE.md
└── results/                 # Experiment-Ergebnisse
```

### Test-First Workflow

1. **Schreibe erst den Test**
   ```bash
   # Test schreiben in tests/test_*.py
   pytest tests/test_new_feature.py -v
   ```

2. **Implementiere die Funktion**
   ```python
   # Code in wend/*/
   # Mit Type Hints, Docstrings, Logging
   ```

3. **Prüfe Checkpoint-Kriterium**
   ```bash
   pytest tests/test_*.py -v
   # Kriterium erfüllt? → Weiter zu Schritt 4
   ```

4. **Dokumentiere (VOR Commit)**
   - Notizzettel in `docs/aktuell/` aktualisieren
   - `docs/PROGRESS.md`: Status ⬜ → ✅, Datum, Notizen
   - NICHT: CLAUDE.md ändern (außer bei Methodik-Änderung)

5. **Commit mit aussagekräftiger Message**
   ```bash
   git add .
   git commit -m "feat(AP2): CP2.1 - SWT Implementierung"
   ```

---

## Task-Tracking

### Drei-Ebenen-System

| Datei | Zweck | Lebensdauer | Wer pflegt? |
|-------|-------|-------------|-------------|
| **CLAUDE.md** | Arbeitsanweisungen, Kriterien (statisch) | Permanent | Bei Methodik-Änderung |
| **docs/PROGRESS.md** | Checkpoint-Status, Logbuch (dynamisch) | Permanent | Nach jedem Checkpoint |
| **docs/aktuell/*.md** | Session-Kontext, TODOs (temporär) | Bis Feature fertig | Während Arbeit |

### Wann Notizzettel anlegen?

**Immer wenn:**
- Neues Arbeitspaket (AP) startest
- Nach Pause/Unterbrechung zurückkommst
- Checkpoint umfangreicher ist (>1h Arbeit)
- Mehrere Teilaufgaben parallel laufen

**Template für Notizzettel:**

```markdown
# [AP-Nummer] [Titel] - [Datum]

## Status
- Aktiver Checkpoint: CPx.y
- Nächster: CPx.z

## Kontext
- Warum arbeite ich daran?
- Was ist die Herausforderung?

## Offene Aufgaben
- [ ] Schritt 1
- [ ] Schritt 2

## Entscheidungen/Erkenntnisse
- [Timestamp] Entscheidung: ...
- [Timestamp] Problem: ... → Lösung: ...

## Nächste Session
- Wo weitermachen?
- Was fehlt noch?
```

**Workflow:**
```bash
# 1. Notizzettel erstellen
touch docs/aktuell/ap2-features-2026-01-13.md

# 2. Während Arbeit aktualisieren
# - Checkboxen abhaken
# - Erkenntnisse notieren

# 3. Bei AP-Abschluss archivieren
mv docs/aktuell/ap2-features-2026-01-13.md docs/archiv/

# 4. Check: README in docs/aktuell/ für Details
```

---

## Prüfliste vor Abschluss

**Bevor du einen Checkpoint als erledigt meldest:**

### 1. Funktioniert es?

```bash
# Type Checking
mypy wend/

# Tests
pytest tests/test_*.py -v

# Spezifischer Test
pytest tests/test_generator.py::test_snr_validation -v

# Coverage (optional)
pytest --cov=wend --cov-report=term-missing
```

### 2. Ist es vollständig?

| Prüfen | Bei Problem → |
|--------|---------------|
| Erfüllt Checkpoint-Kriterium? | `docs/PROGRESS.md` prüfen |
| Alle betroffenen Stellen gefunden? | `grep -r "function_name" wend/` |
| Imports aktualisiert? | `__init__.py` Dateien prüfen |
| Kein toter Code? | Ungenutzte Imports/Funktionen entfernen |

### 3. Ist es sauber?

| Prüfen | Bei Problem → |
|--------|---------------|
| Folgt bestehenden Patterns? | Vergleiche mit ähnlichem Code |
| Type Hints vollständig? | `mypy --strict` |
| Docstrings vorhanden? | Alle public Functions/Classes |
| Logging statt print()? | `grep -r "print(" wend/` (sollte leer sein) |
| Keine TODO/FIXME? | Entweder fixen oder Issue erstellen |

### 4. Bei Problemen

```
Type-Fehler?
├── Fehler-Message komplett lesen
├── Betroffene Datei öffnen
└── Type-Definition nachverfolgen

Tests schlagen fehl?
├── pytest -v für Details
├── pytest --pdb für Debugging
└── Einzelnen Test isoliert laufen lassen

In einer Sackgasse?
├── STOPP - nicht weiterwursteln
├── Notizzettel: Problem dokumentieren
├── Gibt es einfacheren Weg?
└── Bei Unsicherheit: Fragen

Mehrere Ansätze möglich?
├── Gibt es bestehendes Pattern im Code?
├── Wenn ja → diesem folgen
├── Wenn nein → einfachsten Ansatz wählen
└── Bei Architektur-Entscheidung: Fragen
```

---

## Dokumentations-Pflicht

**Nach JEDEM abgeschlossenen Checkpoint:**

### 1. Notizzettel aktualisieren

```markdown
## Status
- ✅ CP2.1 abgeschlossen (2026-01-13)
- Nächster: CP2.2

## Erkenntnisse
- SWT mit pywt funktioniert gut
- coif5 Level 5 gibt 74 Features
```

### 2. PROGRESS.md aktualisieren

```markdown
# Beispiel:
| CP2.1 | ✅ | 2026-01-13 | SWT implementiert, 74 Features extrahiert |
```

**Format:**
- Status: ⬜ → ✅
- Datum: YYYY-MM-DD
- Notizen: Was wurde gemacht, Besonderheiten, Probleme gelöst

### 3. Git Commit (nach Bestätigung)

```bash
# Format:
git add .
git commit -m "feat(APx): CPx.y - Kurzbeschreibung"

# Beispiele:
git commit -m "feat(AP1): CP1.8 - Alle Tests bestehen"
git commit -m "feat(AP2): CP2.1 - SWT Implementierung"
git commit -m "fix(AP2): Feature-Extraktion Reihenfolge korrigiert"
git commit -m "test(AP1): SNR-Validierung erweitert"
git commit -m "docs(AP2): Wavelet-Auswahl dokumentiert"
```

### 4. Wann CLAUDE.md ändern?

**NUR bei:**
- Neue Arbeitsweise/Methodik
- Neue Konventionen
- Änderung der Checkpoint-Kriterien
- Neue Tools/Dependencies

**NICHT bei:**
- Status-Updates (→ PROGRESS.md)
- TODOs (→ Notizzettel)
- Code-Änderungen

---

## Arbeitspakete

| AP | Name | Status | Abhängigkeit |
|----|------|--------|--------------|
| AP1 | Event-Generator | ✅ | - |
| AP2 | Feature-Extraktion | ✅ | AP1 |
| AP3 | Novelty Detection | ✅ | AP2 |
| AP4 | Coverage Estimation | ✅ | AP3 |
| AP5 | Evaluation Framework | ✅ | AP4 |
| AP6 | Erweiterungen | ⬜ | AP5 |

**Details:** Siehe `docs/IMPLEMENTIERUNGSPLAN.md`

---

## AP1: Event-Generator ✅

### Checkpoints

| CP | Kriterium | Command |
|----|-----------|---------|
| CP1.1 | Projekt läuft | `pip install -e . && pytest` |
| CP1.2 | 15 Typen generieren Signale | `pytest tests/test_event_types.py` |
| CP1.3 | Länge/Amplitude valide | `pytest tests/test_generator.py -k "length or amplitude"` |
| CP1.4 | Variation funktioniert | `pytest tests/test_generator.py -k "variation"` |
| CP1.5 | SNR korrekt (±1dB) | `pytest tests/test_generator.py -k "snr"` |
| CP1.6 | Verteilungen korrekt | `pytest tests/test_generator.py -k "distribution"` |
| CP1.7 | Reproduzierbar | `pytest tests/test_generator.py -k "seed"` |
| CP1.8 | Alle Tests grün | `pytest tests/` |

### Event-Typen (15)

**IMPULSE (<10ms):** exponential_spike, dirac_approximation, bipolar_pulse  
**TRANSIENT (10-100ms):** damped_oscillation, inrush_current, chirp_burst  
**SHORT (100ms-1s):** trapezoidal, pwm_burst, multi_peak_envelope  
**MEDIUM (1-10s):** sawtooth_train, modulated_sine, step_sequence  
**LONG (>10s):** slow_ramp_hold, periodic_bursts, wandering_baseline

---

## AP2: Feature-Extraktion ✅

**Status:** Abgeschlossen - 106 Tests, 74 Features pro Event

### Checkpoints

| CP | Kriterium | Command |
|----|-----------|---------|
| CP2.1 | SWT funktioniert | `pytest tests/test_swt.py` |
| CP2.2 | 74 Features pro Event | `pytest tests/test_features.py -k "dimension"` |
| CP2.3 | Features normalisiert | `pytest tests/test_features.py -k "normalization"` |
| CP2.4 | Duration-Klassen korrekt | `pytest tests/test_features.py -k "duration"` |
| CP2.5 | Batch-Verarbeitung | `pytest tests/test_features.py -k "batch"` |
| CP2.6 | Performance OK (<100ms/Event) | `pytest tests/test_performance.py` |

**Details:** Siehe Notizzettel in `docs/aktuell/`

---

## Datenstrukturen

```python
# wend/data/event_types.py

from dataclasses import dataclass
from enum import Enum
import numpy as np
import numpy.typing as npt

class DurationClass(Enum):
    """Event duration categories."""
    IMPULSE = "impulse"       # < 10 ms
    TRANSIENT = "transient"   # 10-100 ms
    SHORT = "short"           # 100 ms - 1 s
    MEDIUM = "medium"         # 1-10 s
    LONG = "long"             # > 10 s

@dataclass
class Event:
    """Container for a single event with metadata."""
    event_id: int
    type_id: int
    signal: npt.NDArray[np.float64]
    duration_samples: int
    duration_ms: float
    amplitude: float
    snr_db: float
    duration_class: DurationClass
    is_novel: bool = False
```

---

## Wichtige Formeln

```python
# Power-law Verteilung
def powerlaw_probs(n_types: int, alpha: float = 1.5) -> np.ndarray:
    """Generate power-law distributed probabilities.
    
    Args:
        n_types: Number of event types
        alpha: Power-law exponent (higher = more skewed)
        
    Returns:
        Array of probabilities summing to 1
    """
    ranks = np.arange(1, n_types + 1)
    probs = ranks ** (-alpha)
    return probs / probs.sum()

# SNR in dB
def calculate_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculate SNR in decibels."""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

# Rauschen für gewünschtes SNR hinzufügen
def add_noise(signal: np.ndarray, snr_db: float, 
              rng: np.random.Generator) -> np.ndarray:
    """Add white noise to achieve target SNR."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise
```

---

## Claude Code Workflows

### Typische Kommandos

```bash
# Entwicklung
pytest tests/ -v                    # Alle Tests
pytest tests/test_X.py -v           # Spezifischer Test
pytest -k "keyword" -v              # Tests mit Keyword
pytest --cov=wend --cov-report=html # Coverage Report

# Code-Qualität
mypy wend/                          # Type Checking
mypy --strict wend/module.py        # Strict Mode für Datei
black wend/ tests/                  # Formatting
ruff check wend/                    # Linting

# Suchen
grep -r "pattern" wend/             # Text-Suche
grep -r "class.*Event" wend/        # Regex-Suche
find wend/ -name "*.py" -type f     # Dateien finden

# Projekt-Status
git status                          # Geänderte Dateien
git diff                            # Änderungen anzeigen
git log --oneline -10               # Letzte Commits
```

### Bei Fehlern

```bash
# Test fehlgeschlagen
pytest tests/test_X.py::test_func -v  # Einzelnen Test
pytest --pdb                           # Debugger bei Fehler

# Type-Fehler
mypy wend/module.py                    # Spezifische Datei
mypy --show-error-codes wend/          # Mit Error-Codes

# Import-Fehler
python -c "import wend; print(wend.__file__)"  # Package-Location
pip list | grep pywt                            # Dependency check
```

---

## Konventionen

- **Sprache Code:** Englisch (Docstrings, Variable, Funktionen)
- **Sprache Docs:** Deutsch (außer Docstrings im Code)
- **Commit-Messages:** Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`)
- **Branch-Namen:** `ap1/event-generator`, `ap2/features`, `fix/snr-calculation`
- **Test-Dateien:** `test_<module>.py` in `tests/`
- **Config-Dateien:** YAML in `configs/`

---

## Bei Problemen

1. **Prüfe `docs/PROGRESS.md`** für letzten Status
2. **Prüfe Notizzettel** in `docs/aktuell/` für Kontext
3. **Checkpoint-Kriterien sind verbindlich**
4. **Bei Unklarheiten: Frage nach, bevor du implementierst**

---

## Schnellreferenz

| Was? | Wo? |
|------|-----|
| Arbeitsanweisungen | Diese Datei |
| Detaillierter Plan | `docs/IMPLEMENTIERUNGSPLAN.md` |
| Checkpoint-Status | `docs/PROGRESS.md` |
| Session-Kontext | `docs/aktuell/*.md` |
| Architektur | `docs/ARCHITECTURE.md` |
| Event-Typen | `wend/data/event_types.py` |
| Tests | `tests/` |
| Konfiguration | `configs/` |

---

*Letzte Aktualisierung: 2026-01-14*