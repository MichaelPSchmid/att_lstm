# Model Evaluation Results (No Dropout)

> **Ziel:** Vergleich von LSTM-Architekturen mit verschiedenen Attention-Mechanismen und MLP-Baselines
> **Task:** Steering Torque Prediction (normiertes Lenkmoment)
> **Stand:** 2026-02-09
> **Methode:** Bootstrap Resampling (n=1000) + Multi-Seed Training (Seeds: 42, 94, 123)

---

## 1. Ergebnisübersicht

| Modell | Typ | Parameter | R² | Accuracy (%) | RMSE | Inference P95 |
|--------|-----|-----------|-----|-------------|------|---------------|
| M1 MLP Last | MLP (5→64→64→1) | 4,609 | 0.708 ± 0.003 | 70.01 ± 0.10 | 0.0590 ± 0.0002 | 0.07 ms |
| M2 MLP Flat | MLP (250→128→64→1) | 40,449 | 0.783 ± 0.003 | 74.90 ± 0.37 | 0.0509 ± 0.0004 | 0.06 ms |
| M3 Small Baseline | LSTM (64, 3L) | 84,801 | 0.860 ± 0.003 | 82.55 ± 0.18 | 0.0409 ± 0.0003 | 0.78 ms |
| M4 Small + Simple Attn | LSTM+Attn (64, 3L) | 84,866 | 0.854 ± 0.004 | 81.95 ± 0.26 | 0.0418 ± 0.0006 | 0.83 ms |
| M5 Medium Baseline | LSTM (128, 5L) | 597,633 | 0.906 ± 0.005 | 88.15 ± 0.80 | 0.0335 ± 0.0008 | 2.66 ms |
| **M6 Medium + Simple Attn** | **LSTM+Attn (128, 5L)** | **597,762** | **0.917 ± 0.002** | **90.04 ± 0.20** | **0.0314 ± 0.0003** | **2.72 ms** |
| M7 Medium + Additive Attn | LSTM+Additive (128, 5L) | 630,529 | 0.910 ± 0.002 | 88.73 ± 0.35 | 0.0328 ± 0.0004 | 3.90 ms |
| M8 Medium + Scaled DP | LSTM+ScaledDP (128, 5L) | 597,633 | 0.909 ± 0.009 | 88.62 ± 1.35 | 0.0330 ± 0.0016 | 2.78 ms |

> *Werte: Mittelwert ± kombinierte Standardabweichung (Bootstrap-Varianz + Seed-Varianz, Law of Total Variance).
> 3 Seeds pro Modell, 1000 Bootstrap-Samples pro Seed. Testset: 220,127 Samples.*

**Ranking nach R²:** M6 (0.917) > M7 (0.910) > M8 (0.909) > M5 (0.906) > M3 (0.860) > M4 (0.854) > M2 (0.783) > M1 (0.708)

---

## 2. Statistische Vergleiche

### 2.1 Paarweise Modellvergleiche

| Vergleich | Kategorie | Δ Acc (pp) | Δ RMSE | Δ R² | Cohen's d | Effekt |
|-----------|-----------|------------|--------|------|-----------|--------|
| M3 → M4 | Baseline vs Attention | −0.56 | +0.0008 | −0.005 | −0.053 | vernachlässigbar |
| M5 → M6 | Baseline vs Attention | +1.87 | −0.0021 | +0.010 | +0.129 | vernachlässigbar |
| M5 → M7 | Baseline vs Attention | +0.56 | −0.0006 | +0.003 | +0.039 | vernachlässigbar |
| M5 → M8 | Baseline vs Attention | +0.73 | −0.0007 | +0.004 | +0.048 | vernachlässigbar |
| M6 → M7 | Attention vs Attention | −1.31 | +0.0015 | −0.007 | −0.099 | vernachlässigbar |
| M6 → M8 | Attention vs Attention | −1.14 | +0.0013 | −0.007 | −0.085 | vernachlässigbar |
| M7 → M8 | Attention vs Attention | +0.16 | −0.0001 | +0.001 | +0.011 | vernachlässigbar |
| M1 → M3 | MLP vs LSTM | +13.28 | −0.0191 | +0.157 | +0.386 | klein |
| M2 → M5 | MLP vs LSTM | +14.49 | −0.0186 | +0.126 | +0.441 | klein |

> *Positive Δ-Werte = Verbesserung von A nach B. Cohen's d berechnet auf gepaarten per-Sample Absolut-Fehlern.
> Schwellen nach Cohen (1988): |d| < 0.2 vernachlässigbar, 0.2–0.5 klein, 0.5–0.8 mittel, > 0.8 groß.
> Permutation Test: 10,000 Permutationen, alle Vergleiche p < 0.01.*

### 2.2 Seed-Stabilität

| Modell | R² (Seed 42) | R² (Seed 94) | R² (Seed 123) | σ_seed |
|--------|-------------|-------------|---------------|--------|
| M1 MLP Last | 0.708 | 0.710 | 0.707 | 0.002 |
| M2 MLP Flat | 0.786 | 0.781 | 0.782 | 0.003 |
| M3 Small Baseline | 0.862 | 0.857 | 0.860 | 0.002 |
| M4 Small + Simple Attn | 0.854 | 0.858 | 0.849 | 0.004 |
| M5 Medium Baseline | 0.905 | 0.911 | 0.903 | 0.005 |
| **M6 Medium + Simple Attn** | **0.919** | **0.917** | **0.916** | **0.001** |
| M7 Medium + Additive Attn | 0.907 | 0.911 | 0.911 | 0.002 |
| M8 Medium + Scaled DP | 0.916 | 0.898 | 0.912 | **0.009** |

> M6 zeigt die geringste Seed-Varianz (σ = 0.001), M8 die höchste (σ = 0.009).

---

## 3. Interpretation

### 3.1 Hauptbefunde

**1. Sequential Modeling ist der entscheidende Faktor.**

Der mit Abstand größte Leistungssprung findet beim Wechsel von MLP auf LSTM statt:
- M1 → M3 (MLP Last → Small LSTM): +0.157 R², Cohen's d = 0.386
- M2 → M5 (MLP Flat → Medium LSTM): +0.126 R², Cohen's d = 0.441

Die explizite Sequenzmodellierung durch LSTMs bringt 13–15 Prozentpunkte mehr Accuracy. Dies bestätigt, dass die temporale Dynamik im Lenkmomentsignal wesentlich ist und nicht durch einfaches Flattening der Zeitschritte erfasst werden kann.

**2. Attention verbessert nur bei ausreichender Modellkapazität.**

Bei kleinen Modellen (64 hidden, 3 Layer) schadet Simple Attention sogar leicht:
- M3 → M4: ΔR² = −0.005, der Attention-Overhead übersteigt den Informationsgewinn

Bei mittleren Modellen (128 hidden, 5 Layer) verbessert Attention konsistent:
- M5 → M6 (Simple): ΔR² = +0.010, bester Attention-Gewinn
- M5 → M7 (Additive): ΔR² = +0.003
- M5 → M8 (Scaled DP): ΔR² = +0.004

**3. Einfache Attention schlägt komplexe Varianten.**

M6 (Simple Attention) übertrifft sowohl M7 (Additive/Bahdanau) als auch M8 (Scaled Dot-Product):

| Attention-Typ | Modell | R² | Extra Params | Inference P95 |
|---------------|--------|-----|-------------|---------------|
| Keine (Baseline) | M5 | 0.906 | — | 2.66 ms |
| **Simple** | **M6** | **0.917** | **+129** | **2.72 ms** |
| Additive (Bahdanau) | M7 | 0.910 | +32,896 | 3.90 ms |
| Scaled Dot-Product | M8 | 0.909 | ±0 | 2.78 ms |

Simple Attention erreicht den höchsten R²-Wert bei minimalem Parameter-Overhead (+129 Parameter) und nahezu identischer Inferenzzeit (+0.06 ms vs Baseline). Die komplexeren Mechanismen (Additive mit +5.5% Parametern, Scaled DP mit höherer Seed-Varianz) liefern keinen Vorteil.

**4. Alle Attention-Unterschiede sind praktisch vernachlässigbar.**

Die Cohen's d-Werte zwischen allen Attention-Varianten liegen unter 0.13 (Schwelle für "klein" wäre 0.2). Die Differenzen sind zwar statistisch signifikant (alle p < 0.01 im Permutation Test), aber bei 220K Testsamples wird nahezu jeder Unterschied signifikant. Die Effektgrößen zeigen: Die Wahl des Attention-Mechanismus hat kaum praktische Relevanz.

### 3.2 Einordnung der Effektgrößen

Cohen's d wurde auf gepaarten per-Sample Absolut-Fehlern berechnet und misst, wie konsistent ein Modell auf Einzelsample-Ebene besser ist als ein anderes. Die Schwellenwerte nach Cohen (1988) stammen aus der Sozialforschung und sind nicht direkt auf Regressionsprobleme übertragbar.

Im vorliegenden Fall gilt:
- **d ≈ 0.4 (MLP → LSTM)**: Konsistente, praktisch relevante Verbesserung. Der R²-Gewinn von 0.13–0.16 entspricht einer deutlich besseren Vorhersage.
- **d ≈ 0.05–0.13 (Baseline → Attention)**: Per-Sample-Verbesserung ist real aber klein. Der aggregierte R²-Gewinn von 0.003–0.010 kann je nach Anwendung relevant sein.
- **d < 0.1 (Attention vs Attention)**: Kein praktisch relevanter Unterschied auf Sample-Ebene.

### 3.3 Empfehlung

**M6 (Medium + Simple Attention)** ist das empfohlene Modell:
- Höchster R² (0.917) und höchste Accuracy (90.04%)
- Geringste Seed-Varianz aller Medium-Modelle (σ = 0.001)
- Minimaler Overhead gegenüber Baseline (+129 Parameter, +0.06 ms)
- Erfüllt Inferenz-Ziel (<10 ms auf CPU) mit großem Spielraum

Falls Attention nicht gewünscht: **M5 (Medium Baseline)** erreicht R² = 0.906 ohne Attention-Mechanismus, bei allerdings höherer Seed-Varianz (σ = 0.005).

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
| Attention | — | Simple: score_i = W·h_i + b |
| Parameter | 84,801 | 84,866 (+65) |
| FLOPs | 4.314M | 4.317M |

### 4.3 Medium LSTM (128 hidden, 5 Layer)

| | M5 Baseline | M6 Simple | M7 Additive | M8 Scaled DP |
|--|-------------|-----------|-------------|--------------|
| Hidden Size | 128 | 128 | 128 | 128 |
| Num Layers | 5 | 5 | 5 | 5 |
| Attention | — | W·h_i + b | v^T·tanh(W·h_i + U·h_j) | (h_i·h_j)/√d |
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
| **Test Samples** | **220,127** (10%) |
| **Training Seeds** | 42, 94, 123 |

### Features (Input, 5-dimensional)

| # | Feature | Beschreibung |
|---|---------|--------------|
| 1 | vEgo | Fahrzeuggeschwindigkeit (m/s) |
| 2 | aEgo | Längsbeschleunigung (m/s²) |
| 3 | steeringAngleDeg | Lenkradwinkel (Grad) |
| 4 | roll | Straßen-Rollwinkel (rad) |
| 5 | latAccelLocalizer | Querbeschleunigung (m/s²) |

### Target (Output)

| Feature | Beschreibung | Bereich |
|---------|--------------|---------|
| steerFiltered | Normiertes Lenkmoment | [−1, 1] |

---

## 6. Methodik

### Bootstrap Confidence Intervals

- **Methode:** Non-parametric Bootstrap Resampling
- **Bootstrap Samples:** 1,000 pro Seed
- **Konfidenzintervall:** 95% (2.5.–97.5. Perzentil)
- **Multi-Seed Aggregation:** Law of Total Variance

  ```
  Var_total = E[Var_bootstrap(seed)] + Var(seed_means)
  ```

  Die kombinierte Unsicherheit umfasst sowohl die Stichprobenvariabilität (Bootstrap) als auch die Variabilität durch unterschiedliche Gewichtsinitialisierungen (Seed-Varianz).

### Paarweise Vergleiche

- **Permutation Test:** 10,000 Permutationen, zweiseitig
- **Cohen's d (paired):** Berechnet auf per-Sample Absolut-Fehlern

  ```
  d = mean(|error_A| - |error_B|) / std(|error_A| - |error_B|)
  ```

  Positiver d-Wert = Modell B hat geringere Fehler als Modell A.

- **Schwellen:** |d| < 0.2 vernachlässigbar, 0.2–0.5 klein, 0.5–0.8 mittel, > 0.8 groß (Cohen, 1988)

### Metriken

| Metrik | Definition | Beschreibung |
|--------|------------|--------------|
| R² | 1 − SS_res / SS_tot | Bestimmtheitsmaß |
| RMSE | √(Σ(y − ŷ)² / n) | Root Mean Squared Error |
| MAE | Σ|y − ŷ| / n | Mean Absolute Error |
| Accuracy | Anteil mit |y − ŷ| < 0.05 | Trefferquote innerhalb Schwelle |

### Inference Benchmarking

- 5 unabhängige Runs pro Modell, 1000 Samples pro Run
- 100 Warmup-Iterationen vor Messung
- Device: CPU (single-thread, `num_threads=1`)

---

## 7. Reproduktion

```bash
# Alle Modelle trainieren (3 Seeds)
python scripts/batch_runner.py train --variant no_dropout

# Alle Modelle evaluieren
python scripts/batch_runner.py evaluate --variant no_dropout

# Bootstrap CIs + Paarweise Vergleiche
python scripts/bootstrap_evaluation.py --n-bootstrap 1000 --n-permutations 10000
```

---

*Generiert am: 2026-02-09*
