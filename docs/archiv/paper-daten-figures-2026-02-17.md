# Paper-Daten und Figures liefern - 2026-02-17

## Status
- **ALLE 6 ARBEITSPAKETE ABGESCHLOSSEN**
- Phase 1 (Daten): KOMPLETT
- Phase 2 (Figures): KOMPLETT

## Kontext
- Fuer die Paper-Ueberarbeitung fehlen R²-Werte, Attention-Weight-CSVs, Prediction-CSVs und 3 PGF/TikZ-Figures
- Scope: NUR Daten und Figures. `paper.tex` wird NICHT angefasst
- Aufgabendefinition: `docs/aktuell/CLAUDE_CODE_TASKS.md`

---

## Bestandsaufnahme (Ist-Zustand)

### Verfuegbare Daten
| Ressource | Pfad | Inhalt |
|-----------|------|--------|
| Evaluation-Ergebnisse | `results/no_dropout/m{1-8}/eval.json` | MSE, RMSE, MAE, R², Accuracy (mean±std, 5 Seeds) |
| Per-Seed Evaluations | `results/no_dropout/m{1-8}/eval_seed{N}.json` | Metriken pro Seed |
| Predictions (Seed 42) | `results/paper/M{1-8}*/seed_42/*_predictions.csv` | sample_idx, sequence_id, y_true, y_pred, abs_error |
| Sequence Metrics (Seed 42) | `results/paper/M{1-8}*/seed_42/*_sequence_metrics.csv` | Per-Sequenz MAE, RMSE, Accuracy |
| Attention Weights (no_dropout) | `attention_weights/M{4,6,7,8}*_seed{N}/attention_epoch_*.npy` | Per-Epoch Snapshots, 5 Seeds |
| Attention Weights (dropout) | `attention_weights/M{4,6,7,8}*_Dropout_seed42/attention_test.npy` | Test-Set gemittelt, nur Seed 42 |
| Bootstrap-Ergebnisse | `results/eval_statistics.json` | Bootstrap CIs, Permutationstests, Cohen's d (per metric) |
| Alte Figures | `figures/backup/*.pgf` | Matplotlib-generierte PGF-Dateien |
| Checkpoints | `lightning_logs/M{1-8}*_seed{N}/version_0/checkpoints/*.ckpt` | Alle 40 Runs vorhanden |

### Bereits vorhandene R²-Werte (Sample-Level, aus eval.json)
| Modell | R² mean | R² std |
|--------|---------|--------|
| M1 | 0.692 | 0.006 |
| M2 | 0.771 | 0.005 |
| M3 | 0.826 | 0.002 |
| M4 | 0.828 | 0.001 |
| M5 | 0.828 | 0.003 |
| M6 | 0.831 | 0.003 |
| M7 | 0.830 | 0.002 |
| M8 | 0.825 | 0.002 |

**Plausibel:** Ranking MLP < LSTM erhalten, Werte niedriger als alte (data-leaking) Werte.

---

## Aufgaben

### Phase 1: Daten (AP 1-3)

- [x] **AP 1: R²-Werte berechnen (Sequence-Level)**
  - **Schritt 1:** Re-Evaluation aller 8 Modelle x 5 Seeds mit `--save-predictions`
    `python scripts/batch_runner.py evaluate --variant no_dropout --save-predictions`
    (oder einzeln per evaluate_model.py falls batch_runner Predictions ueberschreibt)
  - **Schritt 2:** Aus Predictions per-Sequenz R² berechnen:
    R² pro Sequenz = 1 - SS_res/SS_tot (analog zu per-Sequenz RMSE/MAE)
  - **Schritt 3:** Ueber Sequenzen aggregieren (mean), dann mean±std ueber 5 Seeds
  - Skript: Entweder R² in `sequence_level_evaluation.py` ergaenzen oder eigenes Skript
  - Output: `r2_values.csv` (model, r2_mean, r2_std)
  - QA: Werte zwischen 0 und 1, Ranking MLP < LSTM

- [x] **AP 2: Attention-Weight-CSVs exportieren**
  - Datenquellen: `attention_weights/M{4,6,7,8}_*_seed{N}/attention_epoch_*.npy`
  - Vorgehen:
    1. Fuer jedes Modell und jeden Seed: letzte Epoch .npy laden
    2. Ueber alle Seeds mitteln (5 Seeds)
    3. Auf Summe = 1.0 normalisieren
    4. Als CSV exportieren (timestep, weight)
  - Skript: Kurzes Python-Skript oder Notebook-Erweiterung
  - Output: `figures/attention_weights_M{4,6,7,8}.csv`
  - QA: Summe = 1.0, Last-5/10/20-Prozentsaetze gegen Referenztabelle pruefen
  - **Achtung:** no_dropout-Varianten haben KEIN `attention_test.npy`, nur Epoch-Snapshots.
    Muss herausfinden, welcher Epoch der letzte/beste ist (best epoch = checkpoint epoch).

- [x] **AP 3: Prediction-CSVs exportieren**
  - Datenquelle: `results/paper/M{3,5,6}_*/seed_42/*_predictions.csv`
  - Vorgehen:
    1. Predictions fuer M3, M5, M6 laden (Seed 42)
    2. Per-Sequenz RMSE berechnen
    3. Repraesentative Sequenzen waehlen (gut=P10-25, median=P50, schwierig=P75-90)
    4. Zeitreihen extrahieren und als CSV exportieren
  - Output: `figures/prediction_seq_{good,median,difficult}.csv`
  - Format: timestep, ground_truth, M3_pred, M5_pred, M6_pred
  - QA: Werte im Bereich [-1, 1]

### Phase 2: Figures (AP 4-6) -- nach Abschluss von Phase 1

- [x] **AP 4: Attention Weights Plot (.pgf)**
  - Input: CSVs aus AP 2
  - 3 Subplots nebeneinander: M6, M7, M8
  - Horizontale Linie bei 0.02 (Uniform-Referenz)
  - Farben: matlabBlue(0,114,189), matlabOrange(217,83,25), matlabPurple(126,47,142)
  - Erzeugung via matplotlib + scienceplots (wie `generate_figures.py`)
  - Output: `figures/attention_weights_plot.pgf`
  - QA: In LaTeX via `\input{}` einbindbar

- [x] **AP 5: Inference-Accuracy Tradeoff (.pgf)**
  - Daten vollstaendig im Task-Dokument angegeben (kein Pipeline-Zugriff noetig)
  - Scatter-Plot: x=Inference P95 (ms), y=Accuracy (%)
  - Marker: MLP, LSTM Baseline, LSTM+Attention unterschiedlich
  - M3 hervorheben (Pareto-optimal)
  - Erzeugung via matplotlib + scienceplots
  - Output: `figures/inference_accuracy_tradeoff.pgf`
  - QA: In LaTeX via `\input{}` einbindbar

- [x] **AP 6: Prediction Timeseries (.pgf)**
  - Input: CSVs aus AP 3
  - 2-3 Subplots untereinander
  - Linien: Ground Truth (schwarz), M3 (blau), M5 (orange), M6 (gruen)
  - Subplot-Titel mit RMSE-Werten
  - Erzeugung via matplotlib + scienceplots
  - Output: `figures/prediction_timeseries.pgf`
  - QA: In LaTeX via `\input{}` einbindbar

---

## Abhaengigkeiten

```
AP 1 (R²)          --> unabhaengig
AP 2 (Attn Weights) --> unabhaengig
AP 3 (Predictions)  --> unabhaengig

AP 4 (Attn Figure)  --> haengt ab von AP 2
AP 5 (Tradeoff Fig) --> unabhaengig (Daten gegeben)
AP 6 (Pred Figure)  --> haengt ab von AP 3
```

**Optimale Reihenfolge:** AP 1 + AP 2 + AP 3 parallel, dann AP 4 + AP 5 + AP 6.

---

## Offene Fragen / Entscheidungen

1. ~~R² Sample-Level vs. Sequence-Level?~~ **ENTSCHIEDEN: Sequence-Level.**
   Erfordert Predictions fuer alle 5 Seeds. Nur Seed 42 in results/paper/ vorhanden
   -> Re-Evaluation mit `--save-predictions` fuer alle Seeds noetig.

2. ~~Attention Weights: Welche Epoch?~~ **ENTSCHIEDEN: Best-Checkpoint-Epoch.**
   Epoch-Nummer aus Checkpoint-Dateiname ableiten, passenden .npy Snapshot laden.

3. ~~Figure-Format?~~ **ENTSCHIEDEN: .pgf via matplotlib + scienceplots.**

---

## Entscheidungen/Erkenntnisse
- [2026-02-17] Bestandsaufnahme: Alle benoetigten Rohdaten vorhanden, kein Re-Training noetig
- [2026-02-17] R² sample-level bereits in eval.json gespeichert (alle 8 Modelle, 5 Seeds)
- [2026-02-17] Attention Weights: no_dropout hat nur Epoch-Snapshots, dropout hat attention_test.npy
- [2026-02-17] Predictions: Nur Seed 42 in results/paper/ verfuegbar
- [2026-02-17] Entscheidung: Sequence-Level R² (nicht Sample-Level) -> Re-Evaluation noetig
- [2026-02-17] Entscheidung: Attention Weights von Best-Checkpoint-Epoch nehmen
- [2026-02-17] Entscheidung: Figures als .pgf (matplotlib + scienceplots)

## Naechste Session
- Alle APs erledigt. Notizzettel kann nach docs/archiv/ verschoben werden.
- paper.tex kann mit den neuen Daten und Figures aktualisiert werden.

---

*Erstellt am: 2026-02-17*
