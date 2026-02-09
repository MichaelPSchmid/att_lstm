# Evaluation Pipeline Redesign - 2026-02-09

## Status
- Phase 1 (DataModule Fix): KOMPLETT (1a-1e)
- Phase 3 (Evaluation anpassen): KOMPLETT (3a-3b)
- Phase 4 (Statistische Tests): KOMPLETT (4a-4b)
- Naechster: Phase 2 - Training (5 Seeds x 8 Modelle, ~70h)

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
- [x] **3a: Predictions speichern** (2026-02-09)
  - `--save-predictions` ist jetzt Default (neues Flag: `--no-save-predictions`)
  - CSV Spalten: `sample_idx, sequence_id, y_true, y_pred, abs_error`
  - Pfad: `results/{variant}/{model_name}/seed_{seed}/{model_name}_predictions.csv`
  - `save_predictions_csv()` erweitert um optionalen `sequence_ids` Parameter
  - 4 Tests in `TestSavePredictionsCsv`, alle gruen
- [x] **3b: Aggregierte Metriken pro Sequenz berechnen** (2026-02-09)
  - `aggregate_metrics_per_sequence()` in `scripts/shared/metrics.py`
  - Pro Testsequenz: MAE, RMSE, Accuracy (ergibt ~500 unabhaengige Datenpunkte)
  - Gibt (rows, summary) zurueck mit mean/std/median pro Metrik
  - Sequence-Metriken CSV: `{model_name}_sequence_metrics.csv` im selben Verzeichnis
  - Ergebnis in `eval.json` unter `sequence_metrics` key
  - 9 Tests in `TestAggregateMetricsPerSequence`, alle gruen
  - Total: 13 neue Tests + 27 bestehende = 40 Tests gruen

### Phase 4: Neue statistische Tests
- [x] **4a: `scripts/sequence_level_evaluation.py`** (2026-02-09)
  - `bootstrap_ci_sequences()` - Block-Bootstrap auf per-Sequenz-Metriken (vectorisiert)
  - `cohens_d_paired_sequences()` - Gepaarter Cohen's d + Hedge's g (bias-korrigiert)
  - `permutation_test_sequences()` - Sign-Flip Permutationstest (vectorisiert)
  - `multi_seed_sequence_analysis()` - Law of Total Variance ueber Seeds
  - `_compute_seq_metric_arrays()` nutzt `aggregate_metrics_per_sequence` aus shared/metrics.py
  - `run_all_comparisons()` - Alle Paare vergleichen (9 Paare, 3 Metriken)
  - Folgt Muster von `bootstrap_evaluation.py`: Inference live, alle Modelle, multi-seed
  - Metriken: accuracy, rmse, mae (statt accuracy, rmse, r2 bei Sample-Level)
  - 26 Tests in `test_sequence_level_evaluation.py`, alle gruen
- [x] **4b: Output-Formate** (2026-02-09)
  - `results/bootstrap/sequence_level_results_{variant}.json`
  - `results/bootstrap/sequence_level_table_{variant}.tex`
  - `results/bootstrap/sequence_level_table_{variant}.md`
  - Markdown + LaTeX Tabellen fuer Bootstrap CIs und Pairwise Comparisons
  - Comparison-Tabellen enthalten: Cohen's d, Hedge's g, Effektstaerke, p-Werte
  - Total: 39 neue Tests + 27 bestehende = 66 Tests gruen

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
- Phase 2: Training starten (5 Seeds x 8 Modelle)
  - Seeds: [42, 94, 123, 7, 2024]
  - split_seed=0 bleibt fuer alle gleich (identischer Test-Split)
  - Geschaetzte Dauer: ~70h (~3 Tage)
  - Nach jedem Modell: `evaluate_model.py` (speichert Predictions + Seq-Metrics automatisch)
- Danach: `python scripts/sequence_level_evaluation.py` ausfuehren
- Code-Aenderungen Phase 1/3/4 sind komplett und getestet (66 Tests gruen)
