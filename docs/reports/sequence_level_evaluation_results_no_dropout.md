# Sequence-Level Evaluation Results (No Dropout)

> **Ziel:** Vergleich von LSTM-Architekturen mit verschiedenen Attention-Mechanismen und MLP-Baselines
> **Task:** Steering Torque Prediction (normiertes Lenkmoment)
> **Stand:** 2026-02-18
> **Methode:** Multi-Seed Training (Seeds: 7, 42, 94, 123, 231), Sequenz-Level Metriken, Block-Bootstrap (n=1000) fuer stat. Tests
> **Vorgaenger:** `model_evaluation_results_no_dropout.md` (Sample-Level, 3 Seeds, **zurueckgezogen wegen Data Leakage**)

---

## 1. Ergebnisuebersicht

| Modell | Typ | Parameter | Accuracy (%) | RMSE | MAE | R² (seq) | P95 (ms) |
|--------|-----|-----------|-------------|------|-----|----------|----------|
| M1 MLP Last | MLP (5->64->64->1) | 4,609 | 68.98 +/- 0.27 | 0.0559 +/- 0.0003 | 0.0434 +/- 0.0003 | 0.178 +/- 0.011 | 0.08 |
| M2 MLP Flat | MLP (250->128->64->1) | 40,449 | 73.44 +/- 0.47 | 0.0492 +/- 0.0005 | 0.0384 +/- 0.0004 | 0.282 +/- 0.018 | 0.07 |
| M3 Small Baseline | LSTM (64, 3L) | 84,801 | 79.46 +/- 0.35 | 0.0425 +/- 0.0003 | 0.0328 +/- 0.0003 | 0.457 +/- 0.011 | 0.80 |
| M4 Small + Simple Attn | LSTM+Attn (64, 3L) | 84,866 | 79.41 +/- 0.15 | 0.0425 +/- 0.0001 | 0.0328 +/- 0.0001 | 0.459 +/- 0.004 | 0.82 |
| M5 Medium Baseline | LSTM (128, 5L) | 597,633 | 79.63 +/- 0.28 | 0.0423 +/- 0.0002 | 0.0327 +/- 0.0002 | 0.470 +/- 0.006 | 2.81 |
| M6 Medium + Simple Attn | LSTM+Attn (128, 5L) | 597,762 | 79.60 +/- 0.31 | 0.0422 +/- 0.0003 | 0.0326 +/- 0.0002 | 0.469 +/- 0.009 | 2.83 |
| **M7 Medium + Additive Attn** | **LSTM+Additive (128, 5L)** | **630,529** | **79.73 +/- 0.23** | **0.0421 +/- 0.0003** | **0.0325 +/- 0.0002** | **0.470 +/- 0.009** | **4.05** |
| M8 Medium + Scaled DP | LSTM+ScaledDP (128, 5L) | 597,633 | 79.26 +/- 0.17 | 0.0427 +/- 0.0002 | 0.0330 +/- 0.0001 | 0.459 +/- 0.006 | 2.89 |

> *Sequenz-Level Metriken. Mittelwert +/- Standardabweichung ueber 5 Seeds.
> Testset: 500 unabhaengige Fahrsequenzen. Kombinierte Unsicherheit (Bootstrap + Seed) siehe Abschnitt 2.3.*

**Ranking nach Accuracy:** M7 (79.73) > M5 (79.63) > M6 (79.60) > M3 (79.46) > M4 (79.41) > M8 (79.26) > M2 (73.44) > M1 (68.98)

### 1.2 Sample-Level vs Sequence-Level Metriken

| Modell | Accuracy (sample) | Accuracy (seq) | R² (sample) | R² (seq) |
|--------|-------------------|----------------|-------------|----------|
| M1 MLP Last | 70.11 | 68.98 | 0.692 | 0.178 |
| M2 MLP Flat | 74.35 | 73.44 | 0.771 | 0.282 |
| M3 Small Baseline | 80.34 | 79.46 | 0.826 | 0.457 |
| M4 Small + Simple Attn | 80.28 | 79.41 | 0.828 | 0.459 |
| M5 Medium Baseline | 80.55 | 79.63 | 0.828 | 0.470 |
| M6 Medium + Simple Attn | 80.49 | 79.60 | 0.831 | 0.469 |
| M7 Medium + Additive Attn | 80.56 | 79.73 | 0.830 | 0.470 |
| M8 Medium + Scaled DP | 80.17 | 79.26 | 0.825 | 0.459 |

> *Sample-Level: Alle ~221K Testsamples gepoolt. Sequence-Level: R² pro Sequenz, dann Mittelwert ueber 500 Sequenzen.
> Die Diskrepanz entsteht, weil Sequenzen unterschiedlich lang sind (16--550 Samples) und Sample-Level lange Sequenzen uebergewichtet.*

**Kernbeobachtung R²:** Waehrend R² (sample) mit ~0.83 einen guten Fit suggeriert, zeigt R² (seq) mit ~0.47, dass das Modell auf Einzelsequenz-Ebene nur moderate Erklaerungskraft hat. Die Accuracy-basierte Metrik (Schwelle 0.05) ist deutlich optimistischer als R², da sie weniger sensitiv gegenueber grossen Ausreissern innerhalb einer Sequenz ist.

---

## 2. Statistische Vergleiche

### 2.1 Paarweise Modellvergleiche (Sequenz-Ebene)

| Vergleich | Kategorie | Delta Acc (pp) | Delta RMSE | Delta MAE | d(Acc) | d(RMSE) | d(MAE) |
|-----------|-----------|------------|--------|------|--------|---------|--------|
| M3 -> M4 | Baseline vs Attention | -0.05 | -0.0000 | -0.0000 | -0.025 | +0.004 | +0.006 |
| M5 -> M6 | Baseline vs Attention | -0.03 | -0.0002 | -0.0001 | -0.012 | +0.059 | +0.035 |
| M5 -> M7 | Baseline vs Attention | +0.11 | -0.0002\* | -0.0001\* | +0.062 | +0.091 | +0.090 |
| M5 -> M8 | Baseline vs Attention | -0.37\*\*\* | +0.0004\*\*\* | +0.0004\*\*\* | -0.158 | -0.154 | -0.176 |
| M6 -> M7 | Attention vs Attention | +0.13 | -0.0000 | -0.0001 | +0.072 | +0.020 | +0.047 |
| M6 -> M8 | Attention vs Attention | -0.34\*\*\* | +0.0006\*\*\* | +0.0004\*\*\* | -0.180 | -0.251 | -0.252 |
| M7 -> M8 | Attention vs Attention | -0.48\*\*\* | +0.0006\*\*\* | +0.0005\*\*\* | -0.242 | -0.286 | -0.303 |
| M1 -> M3 | MLP vs LSTM | +10.48\*\*\* | -0.0134\*\*\* | -0.0106\*\*\* | +1.240 | +1.123 | +1.081 |
| M2 -> M5 | MLP vs LSTM | +6.19\*\*\* | -0.0069\*\*\* | -0.0057\*\*\* | +0.826 | +0.725 | +0.691 |

> *Positive Delta-Werte = Verbesserung von A nach B. Cohen's d wird pro Metrik berechnet.
> Vorzeichen-Konvention: positiver d-Wert = B besser (hoehere Accuracy, niedrigerer RMSE/MAE).
> Schwellen nach Cohen (1988): |d| < 0.2 vernachlaessigbar, 0.2-0.5 klein, 0.5-0.8 mittel, > 0.8 gross.
> Permutation Test: 10.000 Sign-Flip-Permutationen auf Sequenz-Ebene. Signifikanz: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001.
> Berechnet auf 500 gepaarten Testsequenzen (5 Seeds gemittelt).*

**Detaillierte p-Werte (Permutationstest):**

| Vergleich | p (Accuracy) | p (RMSE) | p (MAE) |
|-----------|-------------|----------|---------|
| M3 -> M4 | 0.583 | 0.925 | 0.904 |
| M5 -> M6 | 0.784 | 0.198 | 0.449 |
| M5 -> M7 | 0.168 | **0.038** | **0.045** |
| M5 -> M8 | **<0.001** | **<0.001** | **<0.001** |
| M6 -> M7 | 0.110 | 0.655 | 0.292 |
| M6 -> M8 | **<0.001** | **<0.001** | **<0.001** |
| M7 -> M8 | **<0.001** | **<0.001** | **<0.001** |
| M1 -> M3 | **<0.001** | **<0.001** | **<0.001** |
| M2 -> M5 | **<0.001** | **<0.001** | **<0.001** |

> *Bemerkenswert: M5 -> M7 zeigt signifikante RMSE/MAE-Verbesserung (p<0.05), obwohl die Accuracy-Differenz nicht signifikant ist. Dies deutet darauf hin, dass Additive Attention die Fehlergroesse (RMSE, MAE) reduziert, ohne die Accuracy-Schwelle (0.05) haeufiger zu unterschreiten.*

### 2.2 Seed-Stabilitaet (5 Seeds)

| Modell | Acc (Seed 7) | Acc (Seed 42) | Acc (Seed 94) | Acc (Seed 123) | Acc (Seed 231) | sigma_seed |
|--------|-------------|-------------|-------------|---------------|---------------|--------|
| M1 MLP Last | 68.59 | 68.76 | 69.33 | 69.12 | 69.11 | 0.30 |
| M2 MLP Flat | 73.52 | 73.72 | 73.95 | 72.56 | 73.43 | 0.53 |
| M3 Small Baseline | 79.97 | 79.22 | 78.96 | 79.70 | 79.46 | 0.40 |
| M4 Small + Simple Attn | 79.38 | 79.65 | 79.45 | 79.41 | 79.18 | 0.17 |
| M5 Medium Baseline | 79.74 | 79.98 | 79.62 | 79.67 | 79.13 | 0.31 |
| M6 Medium + Simple Attn | 79.76 | 80.12 | 79.48 | 79.31 | 79.33 | 0.34 |
| **M7 Medium + Additive Attn** | 79.68 | 80.08 | 79.38 | 79.81 | 79.72 | **0.25** |
| M8 Medium + Scaled DP | 79.34 | 79.22 | 79.01 | 79.18 | 79.52 | 0.19 |

> M7 zeigt die beste Kombination aus Leistung und Stabilitaet. M4 und M8 haben die geringste Seed-Varianz,
> aber bei niedrigerem Accuracy-Niveau.

### 2.3 Unsicherheitszerlegung (Law of Total Variance)

| Modell | sigma_total | sigma_bootstrap | sigma_seed | Anteil Seed-Varianz |
|--------|------------|----------------|-----------|-------------------|
| M1 MLP Last | 0.82 | 0.76 | 0.30 | 14% |
| M2 MLP Flat | 0.86 | 0.68 | 0.53 | 38% |
| M3 Small Baseline | 0.73 | 0.62 | 0.40 | 29% |
| M4 Small + Simple Attn | 0.63 | 0.60 | 0.17 | 7% |
| M5 Medium Baseline | 0.68 | 0.61 | 0.31 | 21% |
| M6 Medium + Simple Attn | 0.69 | 0.60 | 0.34 | 24% |
| M7 Medium + Additive Attn | 0.64 | 0.59 | 0.25 | 15% |
| M8 Medium + Scaled DP | 0.63 | 0.61 | 0.19 | 9% |

> Die Gesamtunsicherheit wird bei den meisten Modellen von der Bootstrap-Varianz (Stichprobenvariabilitaet) dominiert.
> Bei M2 macht die Seed-Varianz 38% aus, was auf Sensitivitaet gegenueber Initialisierung hinweist.

### 2.4 Bootstrap 95%-Konfidenzintervalle

| Modell | Accuracy 95%-CI | RMSE 95%-CI | MAE 95%-CI |
|--------|----------------|-------------|------------|
| M1 MLP Last | [67.38, 70.58] | [0.0536, 0.0582] | [0.0417, 0.0452] |
| M2 MLP Flat | [71.75, 75.13] | [0.0471, 0.0513] | [0.0367, 0.0400] |
| M3 Small Baseline | [78.03, 80.90] | [0.0409, 0.0441] | [0.0316, 0.0341] |
| M4 Small + Simple Attn | [78.19, 80.64] | [0.0411, 0.0440] | [0.0317, 0.0339] |
| M5 Medium Baseline | [78.29, 80.97] | [0.0408, 0.0439] | [0.0315, 0.0338] |
| M6 Medium + Simple Attn | [78.24, 80.96] | [0.0406, 0.0437] | [0.0314, 0.0338] |
| **M7 Medium + Additive Attn** | **[78.47, 81.00]** | **[0.0406, 0.0436]** | **[0.0314, 0.0336]** |
| M8 Medium + Scaled DP | [78.01, 80.50] | [0.0412, 0.0442] | [0.0319, 0.0341] |

> *95%-Konfidenzintervalle (2.5.--97.5. Perzentil) aus Block-Bootstrap (1000 Samples) aggregiert ueber 5 Seeds mittels Law of Total Variance.
> Alle LSTM-Modelle (M3--M8) haben stark ueberlappende CIs, was die geringe praktische Differenz bestaetigt.*

---

## 3. Interpretation

### 3.1 Hauptbefunde

**1. Sequential Modeling ist der entscheidende Faktor.**

Der groesste Leistungssprung findet beim Wechsel von MLP auf LSTM statt:
- M1 -> M3 (MLP Last -> Small LSTM): +10.48 pp Accuracy, d(Acc) = 1.24, d(RMSE) = 1.12, d(MAE) = 1.08 (**gross**)
- M2 -> M5 (MLP Flat -> Medium LSTM): +6.19 pp Accuracy, d(Acc) = 0.83 (**gross**), d(RMSE) = 0.72, d(MAE) = 0.69 (**mittel**)

Dies bestaetigt, dass temporale Dynamik im Lenkmomentsignal wesentlich ist. Der Effekt ist auf Sequenz-Ebene sogar noch deutlicher als auf Sample-Ebene.

**2. Attention liefert keinen praktisch relevanten Mehrwert bei Accuracy -- aber M7 zeigt signifikante RMSE/MAE-Verbesserung.**

Auf Sequenz-Ebene zeigt keine Attention-Variante eine signifikante Verbesserung der **Accuracy** gegenueber der Baseline:

| Vergleich | Delta Accuracy | d(Acc) | d(RMSE) | d(MAE) | p (Acc) | p (RMSE) | p (MAE) |
|-----------|---------------|--------|---------|--------|---------|----------|---------|
| M5 -> M6 (Simple) | -0.03 pp | -0.012 | +0.059 | +0.035 | 0.78 | 0.20 | 0.45 |
| M5 -> M7 (Additive) | +0.11 pp | +0.062 | +0.091 | +0.090 | 0.17 | **0.038** | **0.045** |
| M5 -> M8 (Scaled DP) | -0.37 pp | -0.158 | -0.154 | -0.176 | <0.001 | <0.001 | <0.001 |

M7 (Additive) ist bei Accuracy nicht signifikant besser, zeigt aber **signifikante RMSE/MAE-Reduktion** (p<0.05). Dies bedeutet: Additive Attention verringert die mittlere Fehlergroesse, ohne dass mehr Samples unter die Accuracy-Schwelle (0.05) fallen. Der Effekt ist jedoch mit d(RMSE) = 0.091 und d(MAE) = 0.090 **vernachlaessigbar** (|d| < 0.2) und die 95%-CIs ueberlappen vollstaendig. M8 (Scaled DP) ist signifikant **schlechter** als die Baseline ueber alle Metriken (p < 0.001, d(Acc) = -0.16, d(RMSE) = -0.15, d(MAE) = -0.18).

**3. Modellkapazitaet Small vs Medium spielt kaum eine Rolle.**

Anders als erwartet bringt die 7-fache Parametererhoehung von Small (85K) auf Medium (598K) nur minimalen Gewinn:

| | Small (M3) | Medium (M5) | Delta |
|--|-----------|------------|-------|
| Accuracy | 79.46% | 79.63% | +0.17 pp |
| RMSE | 0.0425 | 0.0423 | -0.0002 |
| Inference P95 | 0.79 ms | 2.62 ms | +232% |

Der Accuracy-Gewinn ist minimal, die Inferenzzeit verdreifacht sich.

**4. Scaled Dot-Product Attention (M8) schadet konsistent.**

M8 ist das einzige Attention-Modell, das signifikant schlechter als die Baseline abschneidet (p < 0.001). Auch gegenueber den anderen Attention-Varianten zeigt M8 einen konsistent negativen Effekt:

- M6 -> M8: d(Acc) = -0.18, d(RMSE) = -0.25, d(MAE) = -0.25 (**klein**)
- M7 -> M8: d(Acc) = -0.24, d(RMSE) = -0.29, d(MAE) = -0.30 (**klein**)

Der Scaled-DP-Mechanismus scheint fuer diese Zeitreihenaufgabe ungeeignet.

**5. R² (sequence) zeigt moderate Erklaerungskraft.**

Die Sequenz-Level R²-Werte liegen fuer alle LSTM-Modelle bei ~0.46--0.47 und damit deutlich unter den Sample-Level R²-Werten (~0.83). Dieser Unterschied hat zwei Ursachen:

- **Sequenzlaengen-Effekt:** Sample-Level R² wird von langen Sequenzen (bis 550 Samples) dominiert, waehrend kurze Sequenzen (ab 16 Samples) kaum ins Gewicht fallen. Sequence-Level R² gewichtet jede Sequenz gleich.
- **Varianz-Effekt:** Kurze Sequenzen mit geringer Target-Varianz (niedriges SS_tot) erzeugen instabile R²-Werte. Sequenzen mit nahezu konstantem Ziel-Signal werden uebersprungen (SS_tot < 1e-12).

Die MLP-Modelle zeigen den groessten R²-Gap: M1 hat R² (sample) = 0.69 vs R² (seq) = 0.18. Dies deutet darauf hin, dass MLPs auf langen, einfachen Sequenzen akzeptabel praedizieren, aber auf kurzen/schwierigen Sequenzen fast keine Erklaerungskraft haben.

### 3.2 Vergleich mit vorheriger Evaluation (Sample-Level)

Die vorherige Evaluation (`model_evaluation_results_no_dropout.md`) verwendete einen fehlerhaften Sample-Level-Split mit Data Leakage und Sample-Level-Statistik. Die Unterschiede sind erheblich:

| Aspekt | Alt (Sample-Level) | Neu (Sequenz-Level) |
|--------|-------------------|---------------------|
| **Split-Methode** | Sample-Level (random_split) | Sequenz-Level (split_seed=0) |
| **Data Leakage** | Ja (P1) | Nein |
| **Test-Einheit** | 220,127 korrelierte Samples | 500 unabhaengige Sequenzen |
| **Seeds** | 3 (42, 94, 123) | 5 (7, 42, 94, 123, 231) |
| **Bootstrap** | Sample-Resampling | Block-Bootstrap (Sequenzen) |
| **Permutationstest** | Sample-Swap | Sign-Flip auf Sequenz-Ebene |

#### Metrik-Vergleich

| Modell | Accuracy Alt | Accuracy Neu | Drop |
|--------|-------------|-------------|------|
| M1 MLP Last | 70.01% | 68.98% | -1.03 pp |
| M2 MLP Flat | 74.90% | 73.44% | -1.46 pp |
| M3 Small Baseline | 82.55% | 79.46% | -3.09 pp |
| M4 Small + Simple Attn | 81.95% | 79.41% | -2.54 pp |
| M5 Medium Baseline | 88.15% | 79.63% | **-8.52 pp** |
| M6 Medium + Simple Attn | 90.04% | 79.60% | **-10.44 pp** |
| M7 Medium + Additive Attn | 88.73% | 79.73% | **-9.00 pp** |
| M8 Medium + Scaled DP | 88.62% | 79.26% | **-9.36 pp** |

Zentrale Beobachtungen:
- **MLP-Modelle verlieren wenig** (-1 bis -1.5 pp): Sie koennen Leakage kaum nutzen, da sie keine Sequenzinformation verarbeiten.
- **Medium-LSTM-Modelle verlieren massiv** (-8.5 bis -10.4 pp): Sie profitierten am staerksten vom Leakage, weil sie gelernte temporale Muster auf nahezu identische Nachbar-Samples im Testset anwenden konnten.
- **M6 verliert am meisten** (-10.44 pp): Die vorherige "Ueberlegenheit" von Simple Attention war ein Artefakt. Der Attention-Mechanismus konnte die geleakten lokalen Muster besonders gut ausnutzen.

#### Warum waren die alten Ergebnisse verzerrt?

1. **Data Leakage (P1):** Samples aus derselben Fahrsequenz in Train und Test. Bei 10 Hz und Window=50 sind aufeinanderfolgende Samples nahezu identisch. Medium-LSTM mit Attention konnte diese Naehe besonders gut ausnutzen.
2. **Aufgeblaehte Statistik (P4/P6):** 220K korrelierte Samples ergeben winzige Konfidenzintervalle und Schein-Signifikanz. Die effektive Stichprobengroesse war deutlich kleiner.
3. **Asymmetrischer Bias:** Modelle mit mehr Kapazitaet und Attention profitierten ueberproportional vom Leakage, wodurch Attention faelschlicherweise als vorteilhaft erschien.

### 3.3 Empfehlung

**M3 (Small Baseline)** oder **M5 (Medium Baseline)** sind die empfohlenen Modelle:

| Kriterium | M3 Small Baseline | M5 Medium Baseline |
|-----------|-------------------|-------------------|
| Accuracy | 79.46% | 79.63% |
| RMSE | 0.0425 | 0.0423 |
| R² (seq) | 0.457 | 0.470 |
| R² (sample) | 0.826 | 0.828 |
| Parameter | 84,801 | 597,633 |
| Inference P95 | 0.80 ms | 2.81 ms |
| Seed-Stabilitaet | sigma = 0.35 | sigma = 0.28 |

- **M3** bietet das beste Kosten-Nutzen-Verhaeltnis: nahezu identische Accuracy bei 7x weniger Parametern und 3x schnellerer Inferenz.
- **M5** ist marginal besser (+0.17 pp) mit geringerer Seed-Varianz, aber deutlich schwerer und langsamer.
- **Attention-Mechanismen sind nicht empfohlen:** Kein messbarer Vorteil auf Sequenz-Ebene bei zusaetzlicher Komplexitaet und Inferenzzeit.

---

## 4. Modellarchitekturen

### 4.1 MLP Baselines

| | M1 MLP Last | M2 MLP Flat |
|--|-------------|-------------|
| Input | Letzter Zeitschritt (5 Features) | Alle 50 Zeitschritte flattened (250 Features) |
| Hidden Layers | [64, 64] | [128, 64] |
| Parameter | 4,609 | 40,449 |
| FLOPs | 4.480K | 40.256K |
| Zweck | Ablation: Wert temporaler Info | Ablation: Sequenz- vs Flat-Verarbeitung |

### 4.2 Small LSTM (64 hidden, 3 Layer)

| | M3 Baseline | M4 + Simple Attn |
|--|-------------|-------------------|
| Hidden Size | 64 | 64 |
| Num Layers | 3 | 3 |
| Attention | -- | Simple: score_i = W*h_i + b |
| Parameter | 84,801 | 84,866 (+65) |
| FLOPs | 4.314M | 4.317M |

### 4.3 Medium LSTM (128 hidden, 5 Layer)

| | M5 Baseline | M6 Simple | M7 Additive | M8 Scaled DP |
|--|-------------|-----------|-------------|--------------|
| Hidden Size | 128 | 128 | 128 | 128 |
| Num Layers | 5 | 5 | 5 | 5 |
| Attention | -- | W*h_i + b | v^T*tanh(W*h_i + U*h_j) | (h_i*h_j)/sqrt(d) |
| Parameter | 597,633 | 597,762 | 630,529 | 597,633 |
| FLOPs | 30.131M | 30.138M | 31.770M | 30.131M |

---

## 5. Daten

### Dataset

| Parameter | Wert |
|-----------|------|
| Vehicle | HYUNDAI_SONATA_2020 |
| Variant | paper (5001 files) |
| Window Size | 50 Zeitschritte (5 Sekunden @ 10 Hz) |
| Predict Size | 1 |
| Step Size | 1 |
| **Total Samples** | **2,201,265** |
| **Total Sequenzen** | **4,988** |
| **Test-Sequenzen** | **500** (10%) |
| **Test-Samples** | **~221,000** |
| **Split-Seed** | **0** (identisch fuer alle Modelle) |
| **Training Seeds** | 7, 42, 94, 123, 231 |

### Features (Input, 5-dimensional)

| # | Feature | Beschreibung |
|---|---------|--------------|
| 1 | vEgo | Fahrzeuggeschwindigkeit (m/s) |
| 2 | aEgo | Laengsbeschleunigung (m/s^2) |
| 3 | steeringAngleDeg | Lenkradwinkel (Grad) |
| 4 | roll | Strassen-Rollwinkel (rad) |
| 5 | latAccelLocalizer | Querbeschleunigung (m/s^2) |

### Target (Output)

| Feature | Beschreibung | Bereich |
|---------|--------------|---------|
| steerFiltered | Normiertes Lenkmoment | [-1, 1] |

---

## 6. Methodik

### 6.1 Sequenz-Level-Split (Neu)

Im Gegensatz zur vorherigen Sample-Level-Aufteilung erfolgt der Train/Val/Test-Split auf **Sequenz-Ebene**:

1. Jede Fahrsequenz (CSV-Datei) erhaelt eine eindeutige `sequence_id`
2. Sequenzen werden per `split_seed=0` deterministisch in 70/20/10 aufgeteilt
3. **Alle Samples einer Sequenz landen im gleichen Split** -- kein Leakage moeglich
4. Der Split ist identisch fuer alle Seeds und Modelle

Dies behebt Problem P1 (Data Leakage) aus der vorherigen Pipeline.

### 6.2 Bootstrap Confidence Intervals (Block-Bootstrap)

- **Methode:** Non-parametric Block-Bootstrap auf Sequenz-Ebene
- **Resampling-Einheit:** Ganze Sequenzen (nicht einzelne Samples)
- **Bootstrap Samples:** 1,000 pro Seed
- **Konfidenzintervall:** 95% (2.5.--97.5. Perzentil)
- **Multi-Seed Aggregation:** Law of Total Variance

  ```
  Var_total = E[Var_bootstrap(seed)] + Var(seed_means)
  ```

Dies behebt Problem P6 (Sample-Resampling ignoriert Autokorrelation).

### 6.3 Paarweise Vergleiche (Sequenz-Ebene)

- **Sign-Flip Permutationstest:** 10,000 Permutationen auf gepaarten per-Sequenz Metriken
- **Cohen's d (paired):** Wird **pro Metrik** berechnet auf per-Sequenz Metrik-Differenzen
- **Hedge's g:** Bias-korrigierte Version von Cohen's d (ebenfalls pro Metrik)

  ```
  d = mean(metric_B(seq_i) - metric_A(seq_i)) / std(metric_B(seq_i) - metric_A(seq_i))
  ```

  **Vorzeichen-Konvention:**
  - Accuracy: positiver d = B besser (hoeherer Wert). Kein Sign-Flip.
  - RMSE, MAE: positiver d = B besser (niedrigerer Wert). Sign-Flip nach Berechnung.

  Die Tabellen zeigen d(Acc), d(RMSE) und d(MAE) als separate Spalten.

Dies behebt Problem P4 (Autokorrelation bei per-Sample-Statistik).

### 6.4 Metriken

| Metrik | Ebene | Definition |
|--------|-------|------------|
| Accuracy | Per Sequenz | Anteil Samples mit \|y - y_hat\| < 0.05 innerhalb der Sequenz |
| RMSE | Per Sequenz | sqrt(mean((y - y_hat)^2)) innerhalb der Sequenz |
| MAE | Per Sequenz | mean(\|y - y_hat\|) innerhalb der Sequenz |
| R² (sample) | Gesamt | 1 - SS_res/SS_tot ueber alle ~221K Testsamples |
| R² (sequence) | Per Sequenz | 1 - SS_res/SS_tot pro Sequenz, dann Mittelwert (Sequenzen mit SS_tot < 1e-12 ausgeschlossen) |

Aggregation: Mittelwert ueber alle 500 Testsequenzen ergibt den Punkt-Schaetzer. R² (sample) wird einmalig ueber das gesamte Testset berechnet und dient als Vergleichswert.

### 6.5 Inference Benchmarking

- 5 unabhaengige Runs pro Modell, 1000 Samples pro Run
- 100 Warmup-Iterationen vor Messung
- Device: CPU (single-thread, `num_threads=1`)

---

## 7. Behobene Probleme

Diese Evaluation behebt die folgenden identifizierten Probleme der vorherigen Pipeline:

| Problem | Beschreibung | Status |
|---------|-------------|--------|
| **P1 (KRITISCH)** | Data Leakage durch Sample-Level-Split | Behoben: Sequenz-Level-Split |
| **P2 (HOCH)** | Nur 3 Seeds | Behoben: 5 Seeds |
| **P3 (MITTEL)** | Keine sequence_id in Predictions | Behoben: CSV mit sequence_id |
| **P4 (HOCH)** | Permutationstest auf Sample-Ebene | Behoben: Sign-Flip auf Sequenzen |
| **P5 (MITTEL)** | Multi-Seed-Handling durch Mittelung | Behoben: Law of Total Variance |
| **P6 (MITTEL)** | Bootstrap resampelt Samples | Behoben: Block-Bootstrap auf Sequenzen |

---

## 8. Reproduktion

```bash
# Alle Modelle trainieren (5 Seeds, split_seed=0)
python scripts/batch_runner.py train --variant no_dropout

# Alle Modelle evaluieren (speichert Predictions + Sequence-Metrics)
python scripts/batch_runner.py evaluate --variant no_dropout

# Sequence-Level Bootstrap CIs + Paarweise Vergleiche
python scripts/sequence_level_evaluation.py --n-bootstrap 1000 --n-permutations 10000
```

---

*Aktualisiert am: 2026-02-18 (Cohen's d pro Metrik, aktualisierte d-Werte aus Notebook-Rerun, Signifikanz auf alle Metriken)*
