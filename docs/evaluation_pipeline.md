# Evaluationspipeline — Steering Torque Prediction

## 1. Bestandsaufnahme

### Datensatz
- **Quelle:** commaSteeringControl, Hyundai Sonata 2020
- **Varianten:**
  - "paper": 5.001 Fahrsequenzen
  - "full": ~20.999 Fahrsequenzen
- **Aktuell verwendet:** "paper" (5.001 Sequenzen)
- **Abtastrate:** 10 Hz
- **Sequenzlänge:** ~60 Sekunden (~600 Samples pro Sequenz)
- **Gesamtsamples:** ~3 Millionen
- **Testsamples aktuell:** 220.127 (10%)
- **Normalisierung:** Lenkmoment auf [-1, 1] normalisiert

### Aktueller Train/Test-Split
- 70% Train / 20% Validation / 10% Test
- **Split auf Sample-Ebene** (zufällig, nicht sequenz-aware)
- sequence_ids werden erfasst, aber nicht für Splitting genutzt

### Modelle (8 Architekturen)

| Modell | Architektur | Parameter | Beschreibung |
|--------|-------------|-----------|--------------|
| M1 | MLP (5→64→64→1) | 4.6K | Nur letzter Zeitschritt |
| M2 | MLP (250→128→64→1) | 40K | Alle Zeitschritte flattened |
| M3 | LSTM (64, 3 Layers) | 85K | Small Baseline |
| M4 | LSTM + Simple (64, 3) | 85K | Small + Attention |
| M5 | LSTM (128, 5 Layers) | 598K | Medium Baseline |
| M6 | LSTM + Simple (128, 5) | 598K | Medium + Simple Attention |
| M7 | LSTM + Additive (128, 5) | 631K | Medium + Bahdanau Attention |
| M8 | LSTM + Scaled DP (128, 5) | 598K | Medium + Dot-Product Attention |

**Relevante paarweise Vergleiche:**
- MLP vs. LSTM: M2 vs. M5
- LSTM-Größe: M3 vs. M5
- Attention-Effekt (small): M3 vs. M4
- Attention-Effekt (medium): M5 vs. M6/M7/M8
- Attention-Typen: M6 vs. M7 vs. M8

### Metriken
- **Accuracy:** % Predictions mit |error| ≤ ε=0.05 (5% des normalisierten Bereichs)
- **RMSE**
- **R²**
- **Inference Time:** 95. Perzentil, Single-Sample, CPU
- **Parameter Efficiency:** Accuracy / (|θ| / 10⁵)

### Evaluationscode
- `evaluate_model.py`: Test-Inference → Metriken → eval_seed{X}.json
- `bootstrap_evaluation.py`: Multi-Seed Aggregation, Bootstrap CIs (n=1000), Permutationstests (n=10000)
- **Per-Sample Predictions:** werden berechnet aber verworfen; `--save-predictions` Flag existiert, aktuell nicht genutzt
- **Gespeichert:** nur aggregierte Metriken pro Seed + Bootstrap-Ergebnisse

### Rechenbudget
- Alle 8 Modelle pro Seed: ~14h
- Anzahl Seeds aktuell: 3
- Hardware: NVIDIA 2060 Super, 16GB RAM

---

## 2. Identifizierte Probleme

### P1: Data Leakage durch Sample-Level-Split
- **Schwere: KRITISCH**
- Samples aus derselben Fahrsequenz können in Train und Test landen
- Bei stark autokorrelierten Steering-Torque-Daten ist das ein Informationsleck
- Bisherige Ergebnisse möglicherweise aufgebläht, ggf. unterschiedlich stark pro Architektur
- **Lösung:** Split auf Sequenz-Ebene umstellen, alle Modelle neu trainieren

### P2: Zu wenige Seeds (n=3) für robuste Statistik
- **Schwere: HOCH**
- Cohen's d / Hedge's g mit n=3 extrem instabil auf Seed-Level
- **Lösung:** Auf mindestens 5–7 Seeds erhöhen (realistisch bei ~14h pro Seed für alle Modelle)

### P3: Per-Sample Predictions werden nicht gespeichert
- **Schwere: MITTEL**
- Nur aggregierte Metriken pro Seed; keine Möglichkeit für gepaarte Vergleiche oder Sequenz-Level-Analyse
- `--save-predictions` Flag existiert bereits
- **Lösung:** Per-Sample Predictions mit Sequenz-ID speichern

### P4: Autokorrelation bei Permutationstest & Cohen's d nicht berücksichtigt
- **Schwere: HOCH**
- Der aktuelle Permutationstest (Zeile 513–573 in bootstrap_evaluation.py) permutiert auf **Sample-Ebene** — er tauscht einzelne Predictions zufällig zwischen Modell A und B
- Bei 220k autokorrelierten Samples ist die Annahme der Austauschbarkeit verletzt
- Cohen's d (Zeile 338–366) berechnet auf denselben 220k abhängigen Samples → aufgeblähte Effektstärke
- **Lösung:** Permutation und Effektstärke auf **Sequenz-Ebene** berechnen (aggregierte Fehler pro Fahrsequenz permutieren)

### P5: Multi-Seed-Handling durch Mittelung der Predictions
- **Schwere: MITTEL**
- Bei Multi-Seed werden Predictions gemittelt, dann ein einzelner Permutationstest durchgeführt
- Das kollapst die Seed-Variabilität und unterschätzt die Unsicherheit
- **Lösung:** Seed-Variabilität explizit in die statistische Analyse einbeziehen

### P6: Bootstrap resampelt einzelne Samples (Autokorrelation ignoriert)
- **Schwere: MITTEL**
- Bootstrap (Zeile 84–148) zieht einzelne Samples mit Zurücklegen
- Bei autokorrelierten Zeitreihen führt das zu zu schmalen Konfidenzintervallen
- **Lösung:** Block-Bootstrap auf Sequenz-Ebene (ganze Sequenzen resampeln)

---

## 3. Offene Fragen

- [x] Welche Modelle genau? → M1–M8 (siehe oben)
- [x] Welche Metriken? → Accuracy, RMSE, R², Inference Time, PE
- [x] Abtastrate? → 10 Hz
- [x] Typische Sequenzlänge? → ~60s (~600 Samples)
- [x] Dauer Trainingslauf? → ~14h für alle 8 Modelle pro Seed
- [x] Struktur des Evaluationscodes? → evaluate_model.py + bootstrap_evaluation.py
- [x] Format der gespeicherten Predictions? → Nur aggregiert; per-Sample optional
- [ ] Physikalisches Lenkmoment-Äquivalent von ε=0.05?
- [ ] Wie viele Seeds sind realistisch? → **5 Seeds** (~70h = ~3 Tage)

---

## 4. Geplante Pipeline

### Überblick

Die neue Pipeline besteht aus 4 Phasen:

```
Phase 1: DataModule Fix     → Sequenz-aware Split + sequence_ids laden
Phase 2: Training           → 5 Seeds × 8 Modelle mit neuem Split
Phase 3: Evaluation         → Per-Sample Predictions + Sequenz-ID speichern
Phase 4: Statistische Tests → Sequenz-Level Permutationstest, Cohen's d, Bootstrap
```

### Phase 1: DataModule Fix (Code-Änderungen)

**1a. sequence_ids im DataModule laden**
- `model/data_module.py`: `sequence_ids_*.pkl` mitlesen
- Als Attribut speichern für späteren Zugriff

**1b. Sequenz-Level-Split implementieren**
- Alle einzigartigen sequence_ids sammeln
- Diese IDs in 70/20/10 splitten (Train/Val/Test)
- Alle Samples einer Sequenz landen im selben Split
- **Deterministisch:** Split-Seed explizit setzen und speichern
- Split-Zuordnung (welche Sequenz-IDs in welchem Split) auf Disk persistieren

**1c. Sequenz-ID pro Sample im Dataset verfügbar machen**
- `__getitem__` gibt (X, Y, seq_id) zurück, oder seq_id als separates Mapping

### Phase 2: Training (5 Seeds × 8 Modelle)

- Seeds: [42, 94, 123, 7, 2024]
- Bisherige Seeds (42, 94, 123) werden beibehalten, zwei neue hinzugefügt
- **Wichtig:** Der Daten-Split bleibt über alle Seeds identisch (gleicher Split-Seed)
  - Nur die Modellinitialisierung und ggf. Shuffling ändern sich pro Seed
- Geschätzte Dauer: ~70h (~3 Tage)
- Checkpoints + Config pro Run speichern

### Phase 3: Evaluation (pro Modell/Seed)

**3a. Predictions speichern**
- Für jedes Modell/Seed: Alle Test-Predictions speichern als CSV/NPY:
  - `sample_idx, sequence_id, y_true, y_pred, abs_error`
- `--save-predictions` Flag standardmäßig aktivieren

**3b. Aggregierte Metriken pro Sequenz berechnen**
- Pro Testsequenz: MAE, RMSE, Accuracy über alle Samples der Sequenz
- Ergebnis: ~500 unabhängige Datenpunkte pro Modell/Seed (statt 220k abhängige)

### Phase 4: Statistische Tests (neu)

**Grundprinzip:** Alle Tests operieren auf Sequenz-Level-Metriken (~500 Testsequenzen)

**4a. Sequenz-Level Cohen's d (gepaart)**
```
Für jedes Modellpaar (A, B):
  1. Pro Testsequenz: RMSE_A und RMSE_B berechnen (primäre Metrik)
  2. Differenzen: d_i = RMSE_A_i - RMSE_B_i  (positiv = B besser)
  3. Cohen's d = mean(d) / std(d, ddof=1)
  4. Hedge's g Korrektur (wegen endlicher Stichprobe)
  5. 95% CI für d via Bootstrap
  6. MAE ergänzend im Appendix berechnen
```

**4b. Sequenz-Level Permutationstest (gepaart)**
```
Für jedes Modellpaar (A, B):
  1. Pro Testsequenz: Metrik für A und B
  2. Beobachtete Differenz der Mittelwerte
  3. Permutation: Für jede Sequenz zufällig A/B tauschen
  4. p-Wert aus Permutationsverteilung
  → ~500 unabhängige Einheiten statt 220k abhängige Samples
```

**4c. Sequenz-Level Bootstrap für Konfidenzintervalle**
```
Für jedes Modell:
  1. Pro Testsequenz: Metriken berechnen
  2. Bootstrap: Ganze Sequenzen resampeln (nicht einzelne Samples)
  3. 95% CIs aus Bootstrap-Verteilung
```

**4d. Multi-Seed Aggregation (Option A)**
```
1. Pro Seed: Sequenz-Level Cohen's d, Permutationstest, Bootstrap-CIs separat berechnen
2. Über Seeds aggregieren:
   - Mean und Std der Cohen's d Werte
   - Anteil der Seeds, bei denen der Effekt signifikant ist
   - Zeigt Stabilität der Ergebnisse über Initialisierungen
```

### Zusammenfassung der Vergleiche

| Vergleich | Frage | Modelle |
|-----------|-------|---------|
| MLP vs. LSTM | Bringt sequentielle Modellierung etwas? | M1 vs. M3, M2 vs. M5 |
| LSTM-Größe | Lohnt sich mehr Kapazität? | M3 vs. M5 |
| Attention-Effekt (small) | Hilft Attention bei kleinen Modellen? | M3 vs. M4 |
| Attention-Effekt (medium) | Hilft Attention bei mittleren Modellen? | M5 vs. M6/M7/M8 |
| Attention-Typen | Welcher Mechanismus ist am besten? | M6 vs. M7 vs. M8 |

### Erwartete Outputs

Pro Vergleich:
- Cohen's d (Hedge's g) + 95% CI + Effektstärke-Kategorie
- Permutationstest p-Wert
- Δ Accuracy, Δ RMSE, Δ R² mit CIs
- Aufgeschlüsselt pro Seed + aggregiert

---

## 5. Implementierungsreihenfolge

1. [ ] **DataModule anpassen** (sequence_ids laden, Sequenz-Level-Split)
2. [ ] **Split validieren** (kein Leakage, Sequenz-Verteilung prüfen)
3. [ ] **Training starten** (5 Seeds × 8 Modelle)
4. [ ] **Evaluation anpassen** (Predictions + Sequenz-ID speichern)
5. [ ] **Neue statistische Tests implementieren** (Sequenz-Level)
6. [ ] **Ergebnisse generieren und Paper aktualisieren**

---

## 6. Detaillierte Implementierungsspezifikation (für Claude Code)

### 6.1 Phase 1: DataModule Fix

**Datei: `model/data_module.py`**

Änderung 1 — sequence_ids laden:
- In `setup()` zusätzlich `sequence_ids_*.pkl` laden (Pfad analog zu features/targets)
- Als `self.sequence_ids` speichern (numpy array oder list, gleiche Länge wie features)

Änderung 2 — Sequenz-Level-Split:
- Alle einzigartigen sequence_ids extrahieren
- Diese IDs mit festem Seed (z.B. `split_seed=0`) in 70/20/10 splitten
- Sample-Indices aus den jeweiligen Sequenz-IDs ableiten
- `torch.utils.data.Subset` mit den korrekten Indices erstellen
- **Split-Zuordnung speichern:** JSON-Datei mit `{train_seq_ids: [...], val_seq_ids: [...], test_seq_ids: [...]}`

Änderung 3 — sequence_id pro Sample verfügbar machen:
- Option: Separates Mapping `test_sample_idx → sequence_id` bereitstellen
- Oder: Dataset `__getitem__` gibt Tuple `(X, Y, seq_id)` zurück
  - **Achtung:** Wenn `__getitem__` geändert wird, müssen Training-Loop und Evaluation kompatibel bleiben

Edge Cases:
- Sequenzen mit sehr wenigen Samples (<10) trotzdem als ganze Sequenz behandeln
- Sicherstellen, dass jeder Split mindestens einige Sequenzen enthält
- Split-Seed ist unabhängig vom Trainings-Seed

**Validierung nach Implementierung:**
```python
# Kein Overlap zwischen Splits
assert len(set(train_seq_ids) & set(test_seq_ids)) == 0
assert len(set(train_seq_ids) & set(val_seq_ids)) == 0
assert len(set(val_seq_ids) & set(test_seq_ids)) == 0

# Alle Samples zugeordnet
assert len(train_indices) + len(val_indices) + len(test_indices) == total_samples

# Split-Proportionen prüfen (ungefähr 70/20/10 auf Sequenz-Ebene)
```

### 6.2 Phase 3: Evaluation anpassen

**Datei: `scripts/evaluate_model.py`**

- `--save-predictions` standardmäßig aktivieren
- Output-CSV pro Modell/Seed:
  ```
  sample_idx, sequence_id, y_true, y_pred, abs_error
  ```
- Pfad: `results/{variant}/{model_id}/seed_{seed}/{model_id}_predictions.csv`

### 6.3 Phase 4: Neue statistische Tests

**Neue Datei: `scripts/sequence_level_evaluation.py`**

Input: Predictions-CSVs aller Modelle/Seeds

Kernfunktionen:

```python
def aggregate_per_sequence(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert Predictions pro Sequenz.
    
    Returns:
        DataFrame mit columns: sequence_id, rmse, mae, accuracy, n_samples
    """

def cohens_d_paired_sequences(
    seq_metrics_a: pd.DataFrame,  # Sequenz-Metriken Modell A
    seq_metrics_b: pd.DataFrame,  # Sequenz-Metriken Modell B
    metric: str = "rmse"
) -> dict:
    """Berechnet gepaarten Cohen's d auf Sequenz-Ebene.
    
    Returns:
        {cohens_d, hedges_g, ci_lower, ci_upper, n_sequences, effect_category}
    """

def permutation_test_sequences(
    seq_metrics_a: pd.DataFrame,
    seq_metrics_b: pd.DataFrame,
    metric: str = "rmse",
    n_permutations: int = 10000,
    seed: int = 42
) -> dict:
    """Gepaarter Permutationstest auf Sequenz-Ebene.
    
    Permutiert: Für jede Sequenz zufällig A↔B tauschen.
    
    Returns:
        {observed_diff, p_value, significant}
    """

def bootstrap_ci_sequences(
    seq_metrics: pd.DataFrame,
    metric: str = "rmse",
    n_bootstrap: int = 1000,
    seed: int = 42
) -> dict:
    """Bootstrap-CIs durch Resampeln ganzer Sequenzen.
    
    Returns:
        {mean, std, ci_lower, ci_upper}
    """

def multi_seed_analysis(
    all_seed_results: Dict[int, dict],  # seed -> Vergleichsergebnis
) -> dict:
    """Aggregiert Ergebnisse über Seeds (Option A).
    
    Returns:
        {mean_d, std_d, min_d, max_d, n_significant, n_seeds}
    """
```

Output:
- `results/bootstrap/sequence_level_results.json`
- `results/bootstrap/sequence_level_table.tex`
- `results/bootstrap/sequence_level_table.md`

### 6.4 Relevante Dateien im Repository

| Datei | Änderung |
|-------|----------|
| `model/data_module.py` | sequence_ids laden, Sequenz-Level-Split |
| `scripts/evaluate_model.py` | Predictions + sequence_id speichern |
| `scripts/bootstrap_evaluation.py` | Ggf. anpassen oder durch neue Datei ersetzen |
| `scripts/sequence_level_evaluation.py` | **NEU** — Sequenz-Level-Tests |
| `scripts/shared.py` | Ggf. erweitern (Sequenz-Utilities) |
| `config/base_config.yaml` | split_seed Parameter hinzufügen |
| `preprocess/preprocess_parallel.py` | Nur lesen — Format der sequence_ids verstehen |
