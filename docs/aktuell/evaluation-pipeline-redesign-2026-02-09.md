# Evaluation Pipeline Redesign - 2026-02-09

## Status
- Aktiver Schritt: Planung / Code-Analyse abgeschlossen
- Naechster: Phase 1a - sequence_ids im DataModule laden

## Kontext
- Die bisherige Evaluationspipeline hat **6 identifizierte Probleme** (P1-P6)
- **P1 (KRITISCH):** Data Leakage durch Sample-Level-Split statt Sequenz-Level-Split
- **P2 (HOCH):** Nur 3 Seeds, zu wenig fuer robuste Statistik
- **P3 (MITTEL):** Per-Sample Predictions werden nicht gespeichert (kein sequence_id)
- **P4 (HOCH):** Permutationstest & Cohen's d auf Sample-Ebene (Autokorrelation ignoriert)
- **P5 (MITTEL):** Multi-Seed-Handling durch Mittelung kollapst Seed-Variabilitaet
- **P6 (MITTEL):** Bootstrap resampelt einzelne Samples statt Sequenzen

## Geplante 4 Phasen

### Phase 1: DataModule Fix (Code-Aenderungen)
- [x] **1a: sequence_ids im DataModule laden** (2026-02-09)
  - `model/data_module.py`: Neuer optionaler Parameter `sequence_ids_path`
  - Geladen in `prepare_data()` als `self.sequence_ids` (numpy int64 array)
  - Rueckwaertskompatibel: ohne Pfad bleibt `self.sequence_ids = None`
  - 9 Tests in `tests/test_data_module.py`, alle gruen
  - Verifiziert mit echten Daten: 2.201.265 Samples, 4.988 Sequenzen
- [x] **1b: Sequenz-Level-Split implementieren** (2026-02-09)
  - `setup()` nutzt jetzt `np.random.RandomState(split_seed)` fuer deterministischen Split
  - Sequenz-IDs werden geschuffelt, dann 70/20/10 zugewiesen
  - Sample-Indices per `np.isin()` abgeleitet, `Subset` statt `random_split`
  - `train/val/test_sequence_ids` als sortierte Listen gespeichert
  - Neuer Parameter `split_seed` (default=0), unabhaengig vom Trainings-Seed
  - 10 neue Tests in `TestSequenceLevelSplit`, alle gruen
  - Echte Daten: 3491 train / 997 val / 500 test Sequenzen, 0 Overlap
- [x] **1c: sequence_id pro Sample verfuegbar machen** (2026-02-09)
  - Methode `get_split_sequence_ids(split)` gibt numpy array zurueck
  - Mapping-Ansatz (Option A): kein Eingriff in Training-Loop / `__getitem__`
  - Gibt `sequence_ids[dataset.indices]` zurueck - gleiche Reihenfolge wie DataLoader
  - 5 neue Tests in `TestGetSplitSequenceIds`, alle gruen
- [x] **1d: Split validieren + persistieren** (2026-02-09)
  - Validierung durch Tests abgedeckt (kein Overlap, alle Samples, Proportionen)
  - `save_split_assignment(path)` speichert JSON mit Sequenz-IDs, Seed, Sample-Counts
  - 4 neue Tests in `TestSaveSplitAssignment`, alle gruen
- [x] **1e: `split_seed` in Config + Aufrufer** (2026-02-09)
  - `config/base_config.yaml`: `split_seed: 0` unter `data:` hinzugefuegt
  - Alle 5 Skripte lesen `data_config.get("split_seed", 0)` und geben ihn an DataModule weiter

### Phase 2: Training (5 Seeds x 8 Modelle)
- [ ] Seeds: [42, 94, 123, 7, 2024]
- [ ] Daten-Split bleibt identisch (gleicher split_seed=0)
- [ ] Nur Modellinitialisierung und Shuffling aendern sich pro Seed
- [ ] Geschaetzte Dauer: ~70h (~3 Tage)
- [ ] Checkpoints + Config pro Run speichern

### Phase 3: Evaluation anpassen
- [ ] **3a: Predictions speichern** (`scripts/evaluate_model.py`)
  - `--save-predictions` standardmaessig aktivieren
  - CSV erweitern: `sample_idx, sequence_id, y_true, y_pred, abs_error`
  - Pfad: `results/{variant}/{model_id}/seed_{seed}/{model_id}_predictions.csv`
- [ ] **3b: Aggregierte Metriken pro Sequenz berechnen**
  - Pro Testsequenz: MAE, RMSE, Accuracy
  - Ergebnis: ~500 unabhaengige Datenpunkte statt 220k abhaengige

### Phase 4: Neue statistische Tests
- [ ] **4a: Neue Datei `scripts/sequence_level_evaluation.py`** mit:
  - `aggregate_per_sequence()` - Predictions pro Sequenz aggregieren
  - `cohens_d_paired_sequences()` - Gepaarter Cohen's d auf Sequenz-Ebene + Hedge's g
  - `permutation_test_sequences()` - Gepaarter Permutationstest (Sequenzen tauschen)
  - `bootstrap_ci_sequences()` - Block-Bootstrap (ganze Sequenzen resampeln)
  - `multi_seed_analysis()` - Aggregation ueber Seeds (Option A)
- [ ] **4b: Output-Formate**
  - `results/bootstrap/sequence_level_results.json`
  - `results/bootstrap/sequence_level_table.tex`
  - `results/bootstrap/sequence_level_table.md`

## Analyse des bestehenden Codes

### `model/data_module.py` (Zeilen 43-106)
- `TimeSeriesDataModule.__init__`: Nimmt nur `feature_path`, `target_path`, `batch_size`
- `setup()`: Laed features/targets, macht `torch.utils.data.random_split()` mit 70/20/10
- **Problem:** Kein `sequence_ids`-Pfad, kein deterministischer Split-Seed
- **Aenderung noetig:** sequence_ids laden, Sequenz-Level-Split, Split-Seed

### `scripts/evaluate_model.py`
- `save_predictions_csv()` (Zeile 403-441): Speichert nur `sample_idx, prediction, target, error, abs_error`
- **Fehlt:** `sequence_id` Spalte
- `run_test_evaluation()` (Zeile 588-623): Gibt `(predictions, targets)` zurueck, keine sequence_ids
- `--save-predictions` ist opt-in Flag

### `scripts/bootstrap_evaluation.py`
- `bootstrap_confidence_intervals()` (Zeile 84-148): Resampelt einzelne Samples
- `permutation_test()` (Zeile 513-573): Tauscht einzelne Predictions zwischen A und B
- `compute_cohens_d()` (Zeile 338-366): Per-Sample absolute errors
- **Alles auf Sample-Ebene** -> muss auf Sequenz-Ebene umgestellt werden

### `preprocess/preprocess_parallel.py`
- `sequence_ids` werden bereits beim Preprocessing erzeugt (Zeile 107-149)
- Format: Liste von ints (Index der CSV-Datei = Fahrsequenz-ID)
- Gespeichert als `.pkl` unter `data/prepared_dataset/*/sequence_ids_*.pkl`
- **Existiert bereits fuer "paper" und "full" Variante**

### `config/settings.py`
- `get_preprocessed_paths()` gibt bereits `sequence_ids` Pfad zurueck (Zeile 81)
- Format: `sequence_ids_{window}_{predict}_{step}_{suffix}.pkl`

## Entscheidungen
- [2026-02-09] Option fuer 1c: Mapping-Ansatz (Option A) bevorzugt, da Training-Loop nicht geaendert werden muss
  - Separate Methode `get_test_sequence_ids()` im DataModule
  - Oder: sequence_ids als separates Array neben dem Dataset speichern

## Abhaengigkeiten & Reihenfolge
```
Phase 1 (DataModule) -> Phase 2 (Training) -> Phase 3 (Evaluation) -> Phase 4 (Statistik)
```
- Phase 1 blockiert alles andere (alle Modelle muessen neu trainiert werden!)
- Phase 3 und 4 sind Code-Aenderungen, die parallel zu Phase 2 vorbereitet werden koennen
- Phase 2 ist die zeitintensivste Phase (~70h GPU-Zeit)

## Risiken
- Training-Loop erwartet `(X, Y)` Tuple aus DataLoader -> sequence_id darf das nicht brechen
- `random_split()` hat keinen expliziten Seed -> Reproduzierbarkeit momentan unklar
- ~500 Testsequenzen sind deutlich weniger Datenpunkte als 220k -> Power sinkt, aber Validitaet steigt

## Naechste Session
- Phase 1a beginnen: sequence_ids in DataModule laden
- Tests schreiben fuer den neuen Sequenz-Level-Split
- base_config.yaml um split_seed erweitern
