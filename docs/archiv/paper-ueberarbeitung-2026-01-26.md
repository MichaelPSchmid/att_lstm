# Paper-Überarbeitung: Kapitel 5 (LSTM-Attention Model)

> **Datum:** 2026-01-26
> **Datei:** `docs/paper/05_lstm-attention model.tex`
> **Datenquelle:** `docs/reports/model_evaluation_results.md`

---

## Status

- [ ] Änderungen planen
- [ ] Änderungen umsetzen
- [ ] Review

---

## Geplante Änderungen

### 1. Trainingszeit → Inferenzzeit

**Problem:** Das Paper vergleicht aktuell Trainingszeiten ("approximately six hours", "4-6 hours"), was für die Anwendung irrelevant ist.

**Lösung:** Umstellen auf Inferenzzeit-Vergleich (relevant für Echtzeit-EPS).

**Betroffene Stellen:**
- Zeile 59: "both models requiring approximately six hours to complete training"
- Zeile 59: "During the later stage of the training process (4-6 hours)"
- Zeile 94: "achieved the highest accuracy of 87.53% in a shorter training time"

**Neue Daten (aus Evaluation):**

| Modell | Mean (ms) | P95 (ms) | Overhead vs Baseline |
|--------|-----------|----------|----------------------|
| M3 Medium Baseline | 2.14 | 2.59 | - |
| M4 Medium + Simple Attn | 2.28 | 2.73 | +6.5% |
| M5 Medium + Additive | 2.64 | 3.26 | +23.4% |
| M6 Medium + Scaled DP | 2.27 | 2.70 | +6.1% |

**Kernaussage:** Attention fügt minimalen Overhead hinzu, alle Modelle unter 10ms (100Hz Echtzeit-fähig).

---

### 2. Dropout-Ablation Study einbringen

**Problem:** Das Paper erwähnt Dropout nicht, obwohl eine vollständige Ablation Study existiert.

**Lösung:** Neuen Abschnitt "Regularization Analysis" oder "Dropout Ablation" hinzufügen.

**Daten (aus `docs/reports/model_evaluation_results_dropout.md`):**

| Model | Dropout | Accuracy | Δ Accuracy |
|-------|---------|----------|------------|
| M3 Baseline | 0.0 | **87.81%** | - |
| M3 Baseline | 0.2 | 86.29% | -1.52% |
| M4 Simple Attn | 0.0 | **90.17%** | - |
| M4 Simple Attn | 0.2 | 84.57% | **-5.60%** |
| M5 Additive | 0.0 | **88.35%** | - |
| M5 Additive | 0.2 | 85.39% | -2.96% |
| M6 Scaled DP | 0.0 | **89.80%** | - |
| M6 Scaled DP | 0.2 | 84.47% | **-5.33%** |

**Kernaussagen:**
- Dropout schadet bei allen Modellen (kein Overfitting vorhanden)
- Attention-Mechanismen besonders betroffen (M4: -5.6%, M6: -5.3%)
- Baseline weniger sensitiv (-1.5%)
- Empfehlung: Kein Dropout für diese Aufgabe

**Mögliche Paper-Formulierung:**
> "We conducted an ablation study on dropout regularization (p=0.2). Results show that dropout consistently degrades performance. Attention-augmented models suffered significant degradation (up to 5.6%), suggesting that attention mechanisms are particularly sensitive to dropout in intermediate layers."

**Offene Frage:** Wo im Kapitel einbauen?
- [ ] Als eigener Subsection nach "Model Performance Comparison"
- [ ] Als Teil eines neuen "Ablation Studies" Kapitels
- [ ] Nur kurz erwähnen + Tabelle

---

### 3. Vergleich verschiedener Attention-Mechanismen

**Problem:** Das Paper zeigt nur "Simple Attention", aber es wurden 3 Typen getestet.

**Lösung:** Vergleichstabelle aller Attention-Mechanismen hinzufügen.

**Daten:**

| Attention Type | R² | Accuracy | RMSE | Params | Overhead |
|----------------|-----|----------|------|--------|----------|
| None (Baseline) | 0.903 | 87.81% | 0.0340 | 597,633 | - |
| Simple (W·h+b) | **0.918** | **90.17%** | **0.0313** | 597,762 | +0.02% |
| Additive (Bahdanau) | 0.907 | 88.35% | 0.0333 | 630,529 | +5.5% |
| Scaled Dot-Product | 0.916 | 89.80% | 0.0317 | 597,633 | 0% |

**Kernaussagen:**
- Simple Attention ist am effektivsten (+2.36% Accuracy vs Baseline)
- Additive Attention hat mehr Parameter, aber schlechtere Performance
- Scaled Dot-Product knapp hinter Simple, aber parameter-effizienter

**Offene Frage:**
- [ ] Alle 3 Typen detailliert beschreiben?
- [ ] Nur erwähnen dass Simple am besten ist?
- [ ] Formeln für alle Attention-Typen?

---

### 4. [Weitere Änderung hier eintragen]

**Problem:**

**Lösung:**

**Betroffene Stellen:**

---

## Aktuelle Werte im Paper vs. Neue Evaluation

### Experiment 1 (Small Model: hidden=64, layers=3)

| Metrik | Paper (alt) | Evaluation (neu) |
|--------|-------------|------------------|
| LSTM Accuracy | 85.06% | 82.54% |
| LSTM+Attn Accuracy | 84.82% | 81.91% |
| R² | - | 0.860 / 0.854 |
| RMSE | - | 0.0408 / 0.0418 |

### Experiment 2 (Medium Model: hidden=128, layers=5)

| Metrik | Paper (alt) | Evaluation (neu) |
|--------|-------------|------------------|
| LSTM Accuracy | 86.75% | 87.81% |
| LSTM+Attn Accuracy | 87.53% | **90.17%** |
| R² | - | 0.903 / 0.918 |
| RMSE | - | 0.0340 / 0.0313 |

### Neue Modelle (nicht im Paper)

| Modell | R² | Accuracy | RMSE |
|--------|-----|----------|------|
| M5 Additive Attention | 0.907 | 88.35% | 0.0333 |
| M6 Scaled Dot-Product | 0.916 | 89.80% | 0.0317 |

---

## Notizen

-

---

## Offene Fragen

- Sollen alle 3 Attention-Typen ins Paper aufgenommen werden?
- Welche Metriken sollen primär gezeigt werden (Accuracy, R², RMSE)?
- Sollen die Figures (Training Curves) aktualisiert werden?

---

*Letzte Aktualisierung: 2026-01-26*
