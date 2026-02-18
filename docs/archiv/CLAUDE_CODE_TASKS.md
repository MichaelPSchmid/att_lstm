# Aufgaben fuer Claude Code: Fehlende Daten und Figures liefern

> **Kontext:** Fuer die Ueberarbeitung des Papers `paper.tex` fehlen noch R²-Werte, Attention-Weight-Rohdaten und Prediction-Rohdaten. Ausserdem muessen drei Figures (Attention Weights, Inference-Accuracy Tradeoff, Prediction Timeseries) neu erstellt werden.
>
> **Scope:** Du lieferst NUR Daten und Figures. Du fasst `paper.tex` NICHT an. Keine Textaenderungen, keine Tabellen-Updates im LaTeX.
>
> **Umgebung:** Du hast Zugriff auf das gesamte Projekt-Repository mit allen Skripten, trainierten Modellen und der Evaluationspipeline. Mache dich zuerst mit der Projektstruktur vertraut (`find`, `ls`, `tree`), um Skripte, Modell-Checkpoints und Datenverzeichnisse zu lokalisieren.
>
> **Referenzdateien (im Paper-Verzeichnis):**
> - `sequence_level_evaluation_results_no_dropout.md` -- Neue korrekte Ergebnisse (Accuracy, RMSE, MAE, Inference, Statistik)
> - `attention_analysis.md` -- Attention-Weight-Analyse (Entropie, temporale Profile, Failure Modes)
> - `analyse.md` -- Vollstaendige Abschnitt-fuer-Abschnitt-Analyse

---

## AP 1: R²-Werte berechnen

**Was fehlt:** R²-Werte fuer alle 8 Modelle (M1-M8) auf Sequenz-Level-Basis. Die bestehende Pipeline berechnet Accuracy, RMSE und MAE, aber kein R².

**Vorgehen:**
1. Finde die gespeicherten Predictions der Sequenz-Level-Evaluation (vermutlich CSVs mit `sequence_id`, `y_true`, `y_pred` pro Modell und Seed).
2. Berechne R² = 1 - SS_res / SS_tot, wobei SS_res = sum((y - y_hat)²) und SS_tot = sum((y - y_mean)²).
3. Berechne R² pro Seed, dann Mittelwert und Standardabweichung ueber 5 Seeds (Seeds: 7, 42, 94, 123, 231).
4. Optional: Auch per-Sequenz R² berechnen und mit Block-Bootstrap aggregieren (analog zu Accuracy/RMSE/MAE in der bestehenden Pipeline).

**Hinweis zur Pipeline:** Die bestehende Evaluation laesst sich reproduzieren mit:
```bash
python scripts/batch_runner.py evaluate --variant no_dropout
python scripts/sequence_level_evaluation.py --n-bootstrap 1000 --n-permutations 10000
```
Falls R² nicht direkt ergaenzt werden kann, schreibe ein eigenes Skript, das die gespeicherten Predictions laedt.

**Plausibilitaetspruefung:** Die alten R²-Werte (mit Data Leakage) waren: M1=0.708, M2=0.786, M3=0.862, M5=0.905, M6=0.919, M7=0.907, M8=0.916. Die neuen Werte sollten niedriger sein, aber das Ranking MLP < Small LSTM < Medium LSTM sollte erhalten bleiben.

**Output:** `r2_values.csv` mit Spalten `model,r2_mean,r2_std` fuer alle 8 Modelle.

---

## AP 2: Attention-Weight-Rohdaten exportieren

**Was fehlt:** Exakte gemittelte Attention Weights als numerische Daten fuer PGF-Plot-Generierung.

**Vorgehen:**
1. Finde das Attention-Analyse-Notebook (laut `attention_analysis.md`: `notebooks/attention_analysis.ipynb`).
2. Exportiere die gemittelten Attention Weights pro Modell (M4, M6, M7, M8) als CSV-Dateien. Shape: 50 Werte (Zeitschritte 0-49), gemittelt ueber alle Testsequenzen und 5 Seeds.
3. Falls das Notebook nicht direkt exportiert: Passe es an oder schreibe ein kurzes Skript, das die gespeicherten Attention Weights laedt und mittelt.

**Plausibilitaetspruefung (aus `attention_analysis.md`):**

| Modell | Last 5 (%) | Last 10 (%) | Last 20 (%) | Peak Pos |
|--------|-----------|------------|------------|----------|
| M4 | 70.1 | 88.6 | 97.3 | 49 |
| M6 | 58.6 | 82.1 | 94.5 | 49 |
| M7 | 91.1 | 99.7 | 100.0 | 49 |
| M8 | 10.0 | 20.1 | 40.1 | ~42 |

**Output:** Je eine CSV-Datei pro Modell im `figures/`-Verzeichnis:
- `figures/attention_weights_M4.csv`
- `figures/attention_weights_M6.csv`
- `figures/attention_weights_M7.csv`
- `figures/attention_weights_M8.csv`

Format: Spalten `timestep,weight` (50 Zeilen, Werte summieren sich zu 1.0).

---

## AP 3: Prediction-Rohdaten exportieren

**Was fehlt:** Predictions vs Ground Truth fuer qualitative Timeseries-Visualisierung.

**Vorgehen:**
1. Finde die gespeicherten Predictions der Sequenz-Level-Evaluation (gleiche Dateien wie AP 1).
2. Waehle 2-3 repraesentative Testsequenzen aus:
   - Eine **gute** Sequenz (niedriger RMSE, z.B. Percentil 10-25)
   - Eine **mittlere** Sequenz (RMSE nahe Median)
   - Eine **schwierige** Sequenz (hoher RMSE, z.B. Percentil 75-90)
3. Exportiere fuer jede Sequenz: Zeitachse, Ground Truth, Predictions von M3 (Small Baseline), M5 (Medium Baseline), M6 (Simple Attention).
4. Verwende einen einzelnen Seed (z.B. Seed 42) fuer die Visualisierung.

**Output:** Je eine CSV-Datei pro Sequenz im `figures/`-Verzeichnis:
- `figures/prediction_seq_good.csv`
- `figures/prediction_seq_median.csv`
- `figures/prediction_seq_difficult.csv`

Format: Spalten `timestep,ground_truth,M3_pred,M5_pred,M6_pred`.

---

## AP 4: Figure 2 -- Attention Weights Visualization (PGF/TikZ)

**Aufgabe:** Neue PGF/TikZ-Plots der Attention Weights erstellen (ersetzt die alte Figure 2 im Paper).

**Datenquelle:** CSV-Dateien aus AP 2.

**Inhalt:** 3 Subplots nebeneinander:
1. M6 (Simple Attention) -- zeigt Recency-Kollaps
2. M7 (Additive Attention) -- zeigt extremen Recency-Kollaps
3. M8 (Scaled Dot-Product) -- zeigt Uniform-Kollaps

Jeder Subplot: x-Achse = Zeitschritt (0-49), y-Achse = Attention Weight. Horizontale gestrichelte Linie bei 0.02 (= 1/50, Uniform-Referenz).

**Format:** Standalone `.tex`-Datei mit PGF/TikZ, die per `\input{}` in `paper.tex` eingebunden werden kann. Verwende die Farben aus `paper.tex`:
- `matlabBlue` (RGB 0,114,189)
- `matlabOrange` (RGB 217,83,25)
- `matlabPurple` (RGB 126,47,142)

**Output:** `figures/attention_weights_plot.tex`

---

## AP 5: Figure 3 -- Inference-Accuracy Tradeoff (PGF/TikZ)

**Aufgabe:** Scatter-Plot mit neuen Accuracy- und Inference-Werten erstellen (ersetzt die alte Figure 3).

**Daten (vollstaendig, kein Zugriff auf Pipeline noetig):**

| Modell | Accuracy (%) | Inference P95 (ms) | Typ |
|--------|-------------|-------------------|-----|
| M1 MLP Last | 68.98 | 0.07 | MLP |
| M2 MLP Flat | 73.44 | 0.06 | MLP |
| M3 Small Baseline | 79.46 | 0.79 | LSTM |
| M4 Small + Simple Attn | 79.41 | 0.79 | LSTM+Attn |
| M5 Medium Baseline | 79.63 | 2.62 | LSTM |
| M6 Medium + Simple Attn | 79.60 | 2.68 | LSTM+Attn |
| M7 Medium + Additive Attn | 79.73 | 3.83 | LSTM+Attn |
| M8 Medium + Scaled DP | 79.26 | 3.59 | LSTM+Attn |

**Gestaltung:**
- x-Achse: Inference P95 (ms), y-Achse: Accuracy (%)
- Verschiedene Marker fuer MLP, LSTM Baseline, LSTM+Attention
- M3 visuell hervorheben (z.B. groesserer Marker oder Annotation "Pareto-optimal")
- Der Plot soll zeigen: grosser Sprung MLP->LSTM, kein Accuracy-Gewinn durch Attention, M3 als bestes Kosten-Nutzen-Modell

**Format:** Standalone `.tex`-Datei mit PGF/TikZ. Farben aus `paper.tex`.

**Output:** `figures/inference_accuracy_tradeoff.tex`

---

## AP 6: Figure 4 -- Prediction Timeseries (PGF/TikZ)

**Aufgabe:** Qualitative Prediction-Plots aus neuen Testsequenzen erstellen (ersetzt die alte Figure 4).

**Datenquelle:** CSV-Dateien aus AP 3.

**Inhalt:** 2-3 Subplots untereinander (eine Sequenz pro Subplot):
- x-Achse: Zeitschritt innerhalb der Sequenz
- y-Achse: Normiertes Lenkmoment (steerFiltered, Bereich [-1, 1])
- Linien: Ground Truth (schwarz, durchgezogen), M3 (blau, gestrichelt), M5 (orange, gestrichelt), M6 (gruen, gepunktet)
- Subplot-Titel: "Good Prediction (RMSE=...)", "Median Prediction (RMSE=...)", "Difficult Prediction (RMSE=...)"

**Format:** Standalone `.tex`-Datei mit PGF/TikZ. Farben aus `paper.tex`.

**Output:** `figures/prediction_timeseries.tex`

---

## Zusammenfassung der erwarteten Outputs

| AP | Output-Datei(en) | Beschreibung |
|----|-----------------|-------------|
| 1 | `r2_values.csv` | R² fuer alle 8 Modelle (mean ± std) |
| 2 | `figures/attention_weights_M{4,6,7,8}.csv` | Attention Weights pro Modell (50 Werte) |
| 3 | `figures/prediction_seq_{good,median,difficult}.csv` | Predictions vs Ground Truth |
| 4 | `figures/attention_weights_plot.tex` | PGF/TikZ Figure 2 |
| 5 | `figures/inference_accuracy_tradeoff.tex` | PGF/TikZ Figure 3 |
| 6 | `figures/prediction_timeseries.tex` | PGF/TikZ Figure 4 |

**Reihenfolge:** AP 1-3 (Daten) zuerst, dann AP 4-6 (Figures).

---

## Qualitaetssicherung

1. **R²-Werte:** Prüfe, dass alle Werte zwischen 0 und 1 liegen und das Ranking plausibel ist (MLP < LSTM).
2. **Attention Weights:** Prüfe, dass die Summe pro Modell = 1.0 (± Rundungsfehler) und die Prozentwerte (Last 5/10/20) mit der Tabelle oben uebereinstimmen.
3. **Prediction CSVs:** Prüfe, dass Ground Truth und Predictions im Bereich [-1, 1] liegen.
4. **PGF/TikZ-Figures:** Prüfe, dass jede `.tex`-Datei standalone kompilierbar ist (`pdflatex figures/xxx.tex`).
5. **Keine Aenderungen an `paper.tex`:** Verifiziere am Ende, dass `paper.tex` unveraendert ist.

---

*Erstellt am: 2026-02-17*
