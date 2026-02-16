# Attention Weight Analysis

> **Ziel:** Untersuchen, warum Attention-Mechanismen keinen Mehrwert gegenueber der LSTM-Baseline liefern
> **Bezug:** `sequence_level_evaluation_results_no_dropout.md` (Sequenz-Level Evaluation)
> **Stand:** 2026-02-16
> **Methode:** Analyse der gemittelten Attention Weights ueber alle Testsequenzen (5 Seeds, finale Epochen + Test-Weights)
> **Notebook:** `notebooks/attention_analysis.ipynb`

---

## 1. Uebersicht

Die Sequenz-Level-Evaluation zeigte, dass keine Attention-Variante die LSTM-Baseline signifikant verbessert (M7 Additive: +0.11 pp, n.s.) und Scaled Dot-Product (M8) sogar signifikant schadet (-0.37 pp, p<0.001). Diese Analyse untersucht die gelernten Attention Weights, um die Ursachen zu identifizieren.

### Untersuchte Modelle

| Modell | Attention-Typ | Scoring-Funktion | Parameter |
|--------|--------------|------------------|-----------|
| M4 | Small Simple | score_i = W*h_i + b | 84,866 |
| M6 | Medium Simple | score_i = W*h_i + b | 597,762 |
| M7 | Medium Additive | score_i = v^T * tanh(W*h_i + U*h_j) | 630,529 |
| M8 | Medium Scaled DP | score_i = (h_i * h_j) / sqrt(d) | 597,633 |

### Datengrundlage

- **Attention Weights:** Gemittelt ueber alle Testsequenzen, Shape (50,) pro Epoch
- **Seeds:** 7, 42, 94, 123, 231 (5 Seeds pro Modell)
- **Zeitschritte:** 50 (= 5 Sekunden @ 10 Hz), Index 0 = aeltester, Index 49 = neuester

---

## 2. Ergebnisse

### 2.1 Entropie-Analyse

Normierte Entropie: 0 = Fokus auf einen einzigen Zeitschritt, 1 = komplett uniform (nutzlos).

| Modell | Norm. Entropie | Std (Seeds) | KL(attn\|\|uniform) | Interpretation |
|--------|---------------|-------------|---------------------|----------------|
| M4 Small Simple | 0.6261 | 0.0969 | 1.4625 | Fokussiert |
| M6 Medium Simple | 0.7130 | 0.0319 | 1.1228 | Fokussiert |
| M7 Medium Additive | 0.4503 | 0.0176 | 2.1506 | Stark fokussiert |
| **M8 Medium Scaled DP** | **1.0000** | **0.0000** | **0.0000** | **Komplett uniform** |

**Befund:** M8 zeigt exakt maximale Entropie -- die Attention-Weights sind perfekt uniform (jeder Zeitschritt erhaelt 1/50 = 0.02). Die KL-Divergenz von der Uniformverteilung betraegt praktisch 0. Der Mechanismus differenziert nicht zwischen Zeitschritten.

M4, M6 und M7 zeigen niedrigere Entropie, lernen also ein fokussiertes Pattern.

### 2.2 Temporales Profil

Wo liegt der Attention-Fokus im 50-Schritt-Window?

| Modell | Last 5 (%) | Last 10 (%) | Last 20 (%) | Peak Position |
|--------|-----------|------------|------------|---------------|
| M4 Small Simple | 70.1 | 88.6 | 97.3 | 49 (alle Seeds) |
| M6 Medium Simple | 58.6 | 82.1 | 94.5 | 49 (alle Seeds) |
| M7 Medium Additive | 91.1 | 99.7 | 100.0 | 49 (alle Seeds) |
| M8 Medium Scaled DP | 10.0 | 20.1 | 40.1 | ~42 (variiert) |

> Erwartungswert bei Uniformverteilung: Last 5 = 10%, Last 10 = 20%, Last 20 = 40%.

**Befund:** M4, M6 und M7 fokussieren massiv auf die letzten Zeitschritte, mit Peak immer bei Position 49 (neuester Zeitschritt). M7 (Additive) ist am extremsten: 99.7% des Gewichts liegt in den letzten 10 Schritten. M8 verteilt das Gewicht exakt gleichmaessig (Werte entsprechen Uniform-Erwartung).

### 2.3 Cross-Seed Konsistenz

Cosine Similarity der finalen Attention Weights zwischen allen Seed-Paaren.

| Modell | Mean CosSim | Std | Min |
|--------|------------|-----|-----|
| M4 Small Simple | 0.9711 | 0.0327 | 0.9050 |
| M6 Medium Simple | 0.9980 | 0.0016 | 0.9946 |
| M7 Medium Additive | 0.9966 | 0.0029 | 0.9905 |
| M8 Medium Scaled DP | 1.0000 | 0.0000 | 1.0000 |

**Befund:** Alle Modelle zeigen hohe Cross-Seed Konsistenz (CosSim > 0.97). Die gelernten Patterns sind stabil ueber verschiedene Initialisierungen. Bei M8 ist die perfekte Konsistenz trivial: alle Seeds konvergieren zur gleichen Uniformverteilung.

---

## 3. Diagnose: Zwei Failure Modes

Die Analyse identifiziert zwei distinkte Mechanismen, durch die Attention versagt:

### Failure Mode 1: Kollaps zu Uniform (M8)

| Eigenschaft | Wert |
|-------------|------|
| Betroffenes Modell | M8 (Scaled Dot-Product) |
| Normierte Entropie | 1.0000 |
| KL-Divergenz | ~0 |
| Accuracy vs Baseline | -0.37 pp (signifikant schlechter) |

**Mechanismus:** Das Scaled Dot-Product `score = (h_i * h_T) / sqrt(d)` erzeugt fuer alle Zeitschritte nahezu identische Scores. Nach Softmax-Normalisierung resultiert eine perfekte Uniformverteilung. Der Attention-Output ist damit ein einfacher Mittelwert ueber alle Hidden States -- informationsaermer als der letzte Hidden State allein.

**Warum schadet es?** Der gewichtete Mittelwert aller Hidden States verwaessert die im letzten Zeitschritt konzentrierte Information. Zusaetzlich erhoehen die Attention-Berechnungen die Inferenzzeit (+37%, 2.62 -> 3.59 ms) ohne Nutzen.

### Failure Mode 2: Kollaps zu Recency (M4, M6, M7)

| Eigenschaft | M4 | M6 | M7 |
|-------------|-----|-----|-----|
| Gewicht auf Position 49 | ~30% | ~20% | ~50% |
| Gewicht Last 10 | 88.6% | 82.1% | 99.7% |
| Accuracy vs Baseline | -0.05 pp | -0.03 pp | +0.11 pp |

**Mechanismus:** Die Attention konvergiert darauf, den letzten Zeitschritt (Position 49) am staerksten zu gewichten. Damit approximiert der Attention-Output den letzten Hidden State h_T -- genau jenen Wert, den das LSTM ohne Attention ohnehin als Output verwendet.

**Warum hilft es nicht?** Die Attention lernt im Wesentlichen die Identitaetsfunktion: Sie gibt nahezu denselben Wert zurueck, den die Baseline-LSTM auch ohne Attention verwendet. Der minimale Beitrag aelterer Zeitschritte reicht nicht aus, um einen messbaren Vorteil zu erzeugen.

---

## 4. Interpretation

### Warum lernt Attention keine nuetzliche Gewichtung?

**1. LSTM-Hidden-State ist bereits ausreichend.**

Das LSTM kodiert die relevante temporale Information kumulativ im Hidden State. Der letzte Hidden State h_T enthaelt bereits eine komprimierte Repraesentation der gesamten Sequenz. Attention ueber die Hidden-State-Sequenz ist daher redundant: Die Information, die Attention extrahieren koennte, ist in h_T schon vorhanden.

**2. Aufgabencharakteristik benoetigt keine langreichweitigen Abhaengigkeiten.**

Steering Torque Prediction bei 10 Hz mit 5-Sekunden-Window haengt primaer vom aktuellen Fahrzeugzustand ab (Geschwindigkeit, Lenkwinkel, Beschleunigung). Es gibt wenig langreichweitige Abhaengigkeiten, die Attention besser als das LSTM erfassen koennte. Die dominante Information liegt in den juengsten Zeitschritten -- genau dort, wo das LSTM ohnehin am staerksten ist.

**3. Window-Groesse ist zu kurz fuer Attention-Vorteile.**

Bei 50 Zeitschritten ist die Sequenz kurz genug, dass ein 3- bis 5-Layer LSTM die gesamte relevante Historie erfassen kann, ohne dass ein expliziter Attention-Mechanismus zum selektiven Zugriff auf aeltere Zeitschritte noetig waere.

### Zusammenhang mit Data-Leakage-Ergebnissen

In der alten Evaluation (Sample-Level, mit Data Leakage) erschien Attention vorteilhaft:
- M6 (Simple Attention) erreichte 90.04% vs M5 (Baseline) 88.15%
- Nach Leakage-Korrektur: M6 79.60% vs M5 79.63%

Die vorherige "Ueberlegenheit" war ein Artefakt: Attention konnte die geleakten lokalen Muster (nahezu identische Nachbar-Samples im Testset) besonders gut ausnutzen. Ohne Leakage verschwindet dieser kuenstliche Vorteil vollstaendig.

---

## 5. Fazit

| Attention-Typ | Failure Mode | Effekt auf Performance |
|---------------|-------------|----------------------|
| Scaled Dot-Product (M8) | Uniform-Kollaps | Signifikant negativ (-0.37 pp) |
| Simple Attention (M4, M6) | Recency-Kollaps | Vernachlaessigbar (~0 pp) |
| Additive Attention (M7) | Recency-Kollaps | Vernachlaessigbar (+0.11 pp, n.s.) |

**Empfehlung bleibt unveraendert:** M3 (Small Baseline LSTM) oder M5 (Medium Baseline LSTM) ohne Attention. Die Attention-Mechanismen fuegen Komplexitaet und Inferenzzeit hinzu, ohne die Vorhersagequalitaet zu verbessern.

---

*Generiert am: 2026-02-16*
*Analyse-Notebook: `notebooks/attention_analysis.ipynb`*
