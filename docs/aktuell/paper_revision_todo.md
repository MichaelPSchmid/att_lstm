# Paper Revision TODO

> **Ziel:** Umstellung des Paper-Fokus von Trainingszeit auf Inferenzzeit
> 
> **Datei:** `docs\paper\paper.tex`
> 
> **Stand:** 2026-01-26

---

## Übersicht der Änderungen

| Bereich | Änderungstyp | Priorität |
|---------|--------------|-----------|
| Abstract | Text ersetzen | 🔴 Hoch |
| Contributions | Text ersetzen | 🔴 Hoch |
| Hardware-Angabe | Korrektur | 🟡 Mittel |
| Metriken (IV.D) | Entfernen + Hinzufügen | 🔴 Hoch |
| Table I | Komplett neu | 🔴 Hoch |
| Section V.A-B | Werte aktualisieren | 🔴 Hoch |
| Section V.C | Komplett umschreiben | 🔴 Hoch |
| Section V.D | Entfernen | 🟡 Mittel |
| Section V.E | Neu erstellen | 🟡 Mittel |
| Figure 4 | Komplett neu | 🔴 Hoch |
| Figure 5 | Erweitern | 🟡 Mittel |
| Conclusion | Aktualisieren | 🔴 Hoch |

---

## Aktuelle Experimentelle Ergebnisse (Referenz)

Diese Werte aus `model_evaluation_results.md` sind die Ground Truth:

```
| Model                      | Params  | Accuracy | R²    | RMSE   | Inference P95 |
|----------------------------|---------|----------|-------|--------|---------------|
| M1 Small Baseline          | 84,801  | 82.57%   | 0.860 | 0.0408 | 1.11 ms       |
| M2 Small + Simple Attn     | 84,866  | 81.50%   | 0.850 | 0.0423 | 1.16 ms       |
| M3 Medium Baseline         | 597,633 | 87.84%   | 0.905 | 0.0338 | 2.40 ms       |
| M4 Medium + Simple Attn    | 597,762 | 90.25%   | 0.919 | 0.0311 | 2.44 ms       |
| M5 Medium + Additive Attn  | 630,529 | 88.34%   | 0.907 | 0.0332 | 2.88 ms       |
| M6 Medium + Scaled DP      | 597,633 | 88.17%   | 0.907 | 0.0334 | 2.46 ms       |
```

Dropout-Ergebnisse aus `model_evaluation_results_dropout.md`:

```
| Model                      | No Dropout | Dropout=0.2 | Δ Accuracy |
|----------------------------|------------|-------------|------------|
| M1 Small Baseline          | 82.57%     | 80.49%      | -2.08%     |
| M2 Small + Simple Attn     | 81.50%     | 80.07%      | -1.43%     |
| M3 Medium Baseline         | 87.84%     | 86.29%      | -1.55%     |
| M4 Medium + Simple Attn    | 90.25%     | 84.31%      | -5.94%     |
| M5 Medium + Additive Attn  | 88.34%     | 85.39%      | -2.95%     |
| M6 Medium + Scaled DP      | 88.17%     | 85.23%      | -2.94%     |
```

---

## 1. Abstract

### 1.1 Trainingszeit-Aussage ersetzen

- [x] **Zeile ~12-13 in Abstract** ✅ (2026-01-26)

**Suchen:**
```latex
showing that simpler model structures achieve comparable performance with significantly reduced training time (\qty{79}{\percent} reduction).
```

**Ersetzen durch:**
```latex
demonstrating that all proposed models achieve inference times below \qty{3}{\milli\second} on CPU, enabling real-time deployment at over \qty{100}{\hertz} for resource-constrained automotive embedded systems.
```

### 1.2 Accuracy-Wert korrigieren

- [x] **Zeile ~8 in Abstract** ✅ (2026-01-26)

**Suchen:**
```latex
Simple Attention achieves the highest prediction accuracy of \qty{89.71}{\percent}, outperforming the baseline LSTM (\qty{84.66}{\percent}) by approximately \qty{5}{\percent}.
```

**Ersetzen durch:**
```latex
Simple Attention achieves the highest prediction accuracy of \qty{90.25}{\percent}, outperforming the baseline LSTM (\qty{87.84}{\percent}) by \qty{2.41}{\percent}.
```

---

## 2. Section I: Introduction

### 2.1 Contribution 3 aktualisieren

- [x] **Im itemize-Block der Contributions** ✅ (2026-01-26)

**Suchen:**
```latex
\item We provide a comprehensive analysis of the trade-off between model complexity and prediction accuracy, showing that simpler models can achieve comparable performance with significantly reduced training time.
```

**Ersetzen durch:**
```latex
\item We provide a comprehensive analysis of the trade-off between model complexity, prediction accuracy, and inference efficiency, demonstrating that attention-enhanced models achieve superior accuracy while maintaining real-time capability ($< \qty{3}{\milli\second}$ inference on CPU) suitable for automotive embedded systems.
```

---

## 3. Section IV: Experimental Setup

### 3.1 Hardware korrigieren (Section IV.C)

- [x] **In "Implementation Details"** ✅ (2026-01-26)

**Suchen:**
```latex
NVIDIA GeForce RTX 3070 Ti GPU with 8~GB VRAM
```

**Ersetzen durch:**
```latex
NVIDIA GeForce RTX 2060 Super GPU with 8~GB VRAM
```

### 3.2 Metriken entfernen (Section IV.D)

- [x] **TCR-Definition entfernen** ✅ (2026-01-26)

**Entfernen (komplett):**
```latex
\textbf{Time Cost Ratio (TCR):} The ratio of training time for a complex model to reach the same accuracy as a simpler model:
\begin{equation}
    \text{TCR} = \frac{T_{\text{complex}}}{T_{\text{simple}}}
    \label{eq:tcr}
\end{equation}
```

- [x] **LE-Definition entfernen** ✅ (2026-01-26)

**Entfernen (komplett):**
```latex
\textbf{Learning Efficiency (LE):} The rate of accuracy improvement per hour:
\begin{equation}
    \text{LE} = \frac{\text{Acc}_{t_2} - \text{Acc}_{t_1}}{t_2 - t_1}
    \label{eq:le}
\end{equation}
```

### 3.3 Neue Metrik hinzufügen (Section IV.D)

- [x] **Nach RMSE-Definition einfügen:** ✅ (2026-01-26)

```latex
\textbf{Inference Time:} The 95th percentile of single-sample prediction time on CPU, ensuring consistent real-time performance:
\begin{equation}
    T_{\text{inf}} = P_{95}(t_1, t_2, \ldots, t_n)
    \label{eq:inference}
\end{equation}
```

---

## 4. Section V.A: Baseline LSTM Performance

### 4.1 Werte aktualisieren

- [x] **Ersten Absatz anpassen** ✅ (2026-01-26)

**Suchen:**
```latex
The baseline LSTM model with hidden size 64 and 3 layers achieves \qty{84.66}{\percent} accuracy and RMSE of 0.00116 on the test set.
```

**Ersetzen durch:**
```latex
The small baseline LSTM model (M1) with hidden size 64 and 3 layers achieves \qty{82.57}{\percent} accuracy and RMSE of 0.0408 on the test set.
```

- [x] **Zweiten Absatz anpassen** ✅ (2026-01-26)

**Suchen:**
```latex
Increasing model complexity (hidden size 128, 5 layers) yields only marginal improvement to \qty{86.75}{\percent}, despite a 7-fold increase in model parameters (from 0.339~MB to 2.391~MB).
```

**Ersetzen durch:**
```latex
Increasing model complexity to the medium configuration (M3, hidden size 128, 5 layers) improves accuracy to \qty{87.84}{\percent}, with a 7-fold increase in model parameters (from 85K to 598K).
```

---

## 5. Section V.B: Impact of Attention Mechanisms

### 5.1 Table I komplett ersetzen

- [x] **Gesamte Tabelle ersetzen** ✅ (2026-01-26)

**Suchen (gesamte Tabelle):**
```latex
\begin{table}[t]
    \centering
    \caption{Performance Comparison of LSTM Models with Different Attention Mechanisms}
    \label{tab:attention_comparison}
    \begin{tabular}{lcc}
        \toprule
        \textbf{Model} & \textbf{Accuracy (\%)} & \textbf{RMSE} \\
        \midrule
        Baseline LSTM & 84.66 & 0.00116 \\
        LSTM + Simple Attention & \textbf{89.71} & \textbf{0.00098} \\
        LSTM + Additive Attention & 88.82 & 0.00101 \\
        LSTM + Scaled Dot-Product Attention & 88.65 & 0.00101 \\
        \bottomrule
    \end{tabular}
\end{table}
```

**Ersetzen durch:**
```latex
\begin{table}[t]
    \centering
    \caption{Performance Comparison of LSTM Models with Different Attention Mechanisms}
    \label{tab:attention_comparison}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Model} & \textbf{Params} & \textbf{Acc. (\%)} & \textbf{RMSE} & \textbf{Inf. (ms)} \\
        \midrule
        M1 Small Baseline & 85K & 82.57 & 0.0408 & 1.11 \\
        M2 Small + Simple Attn & 85K & 81.50 & 0.0423 & 1.16 \\
        \midrule
        M3 Medium Baseline & 598K & 87.84 & 0.0338 & 2.40 \\
        \textbf{M4 Medium + Simple Attn} & 598K & \textbf{90.25} & \textbf{0.0311} & 2.44 \\
        M5 Medium + Additive Attn & 631K & 88.34 & 0.0332 & 2.88 \\
        M6 Medium + Scaled DP Attn & 598K & 88.17 & 0.0334 & 2.46 \\
        \bottomrule
    \end{tabular}
\end{table}
```

### 5.2 Text nach Tabelle aktualisieren

- [x] **Absatz nach Table I** ✅ (2026-01-26)

**Suchen:**
```latex
The LSTM with Simple Attention achieves the highest accuracy (\qty{89.71}{\percent}) and lowest RMSE (0.00098).
```

**Ersetzen durch:**
```latex
The medium LSTM with Simple Attention (M4) achieves the highest accuracy (\qty{90.25}{\percent}) and lowest RMSE (0.0311). Notably, attention mechanisms do not improve performance for small models: M2 (\qty{81.50}{\percent}) underperforms the baseline M1 (\qty{82.57}{\percent}), suggesting that small architectures lack sufficient representational capacity to benefit from attention weighting.
```

---

## 6. Section V.C: Model Complexity Analysis

### 6.1 Komplette Section ersetzen

- [x] **Gesamte Section V.C ersetzen** ✅ (2026-01-26)

**Suchen (von `\subsection{Model Complexity Analysis}` bis zur nächsten `\subsection`):**
- Alles inkl. Table II (Simple vs Complex)
- Alles inkl. Table III (TCR Analysis)  
- Alles inkl. Figure 4 (alte Version)

**Ersetzen durch:**
```latex
\subsection{Inference Efficiency Analysis}

For deployment in automotive embedded systems, inference time is critical. \cref{tab:inference} summarizes the computational efficiency of all models.

\begin{table}[t]
    \centering
    \caption{Inference Efficiency on CPU (Intel Core i7)}
    \label{tab:inference}
    \begin{tabular}{lccc}
        \toprule
        \textbf{Model} & \textbf{Inf. P95 (ms)} & \textbf{Max Hz} & \textbf{Attn. Overhead} \\
        \midrule
        M1 Small Baseline & 1.11 & 900 & -- \\
        M2 Small + Simple & 1.16 & 862 & +4.5\% \\
        \midrule
        M3 Medium Baseline & 2.40 & 417 & -- \\
        M4 Medium + Simple & 2.44 & 410 & +1.7\% \\
        M5 Medium + Additive & 2.88 & 347 & +20.0\% \\
        M6 Medium + Scaled DP & 2.46 & 407 & +2.5\% \\
        \bottomrule
    \end{tabular}
\end{table}

All models comfortably exceed the \qty{100}{\hertz} real-time requirement for EPS systems. The computational overhead of attention mechanisms varies significantly: Simple Attention adds only \qty{1.7}{\percent} latency while providing the largest accuracy gain (\qty{+2.41}{\percent}), whereas Additive Attention increases latency by \qty{20}{\percent} for a smaller accuracy improvement (\qty{+0.50}{\percent}).

\cref{fig:inference_tradeoff} visualizes the accuracy-latency trade-off across all models.

%%% NEW Figure 4: Accuracy vs Inference Time
\begin{figure}[t]
    \centering
    \begin{tikzpicture}
        \begin{axis}[
            width=0.95\columnwidth,
            height=5.5cm,
            xlabel={Inference Time P95 (ms)},
            ylabel={Accuracy (\%)},
            xmin=0.8, xmax=3.2,
            ymin=80, ymax=92,
            grid=major,
            grid style={dashed, gray!30},
            legend pos=south east,
            legend style={font=\scriptsize},
        ]
        
        % Small Models
        \addplot[only marks, mark=o, mark size=3pt, color=matlabBlue] 
            coordinates {(1.11, 82.57) (1.16, 81.50)};
        \addlegendentry{Small (85K)}
        
        % Medium Baseline
        \addplot[only marks, mark=square*, mark size=3pt, color=matlabGreen] 
            coordinates {(2.40, 87.84)};
        \addlegendentry{Medium Baseline}
        
        % Medium + Attention
        \addplot[only marks, mark=triangle*, mark size=4pt, color=matlabOrange] 
            coordinates {(2.44, 90.25) (2.88, 88.34) (2.46, 88.17)};
        \addlegendentry{Medium + Attention}
        
        % Labels
        \node[font=\scriptsize, anchor=south] at (axis cs:1.11,82.57) {M1};
        \node[font=\scriptsize, anchor=north] at (axis cs:1.16,81.50) {M2};
        \node[font=\scriptsize, anchor=east] at (axis cs:2.40,87.84) {M3};
        \node[font=\scriptsize, anchor=south] at (axis cs:2.44,90.25) {M4};
        \node[font=\scriptsize, anchor=west] at (axis cs:2.88,88.34) {M5};
        \node[font=\scriptsize, anchor=north] at (axis cs:2.46,88.17) {M6};
        
        \end{axis}
    \end{tikzpicture}
    \caption{Trade-off between prediction accuracy and inference time. M4 (Simple Attention) achieves the highest accuracy with minimal latency overhead.}
    \label{fig:inference_tradeoff}
\end{figure}

The choice between small and medium models presents a clear trade-off: small models offer approximately \qty{2}{\times} faster inference (\qty{\sim 1}{\milli\second} vs. \qty{\sim 2.5}{\milli\second}) but sacrifice approximately 8 percentage points in accuracy.
```

---

## 7. Section V.D: Window Size - ENTFERNEN

- [x] **Gesamte Section V.D entfernen** ✅ (2026-01-26)

**Entfernen (komplett):**
```latex
\subsection{Impact of Window Size}

The window size significantly affects model performance, as shown in \cref{tab:window_size}.
Increasing the window size from 15 to 50 time steps improves accuracy by approximately \qty{5}{\percent}.

%%% Table: Window Size
\begin{table}[t]
    \centering
    \caption{Impact of Window Size on Model Performance}
    \label{tab:window_size}
    \begin{tabular}{ccc}
        \toprule
        \textbf{Window Size} & \textbf{Accuracy (\%)} & \textbf{RMSE} \\
        \midrule
        15 (1.5s) & 83.86 & 0.00119 \\
        50 (5.0s) & 88.72 & 0.00104 \\
        \bottomrule
    \end{tabular}
\end{table}

Larger windows capture more temporal context, enabling the model to learn longer-term dependencies in vehicle dynamics.
At 10~Hz sampling frequency, a window of 50 time steps corresponds to 5 seconds of driving data, which provides sufficient context for predicting steering torque requirements.
```

---

## 8. Section V.E: Dropout Ablation - NEU ERSTELLEN

- [x] **Neue Section nach V.C (bzw. alter V.D Position) einfügen** ✅ (2026-01-26)

**Einfügen:**
```latex
\subsection{Effect of Dropout Regularization}

To investigate whether dropout regularization could improve generalization, we trained all model variants with dropout probability $p=0.2$ applied to LSTM layers. \cref{tab:dropout} presents the results.

\begin{table}[t]
    \centering
    \caption{Effect of Dropout ($p=0.2$) on Model Performance}
    \label{tab:dropout}
    \begin{tabular}{lccc}
        \toprule
        \textbf{Model} & \textbf{No Dropout} & \textbf{Dropout} & \textbf{$\Delta$} \\
        \midrule
        M1 Small Baseline & 82.57\% & 80.49\% & -2.08\% \\
        M2 Small + Simple & 81.50\% & 80.07\% & -1.43\% \\
        M3 Medium Baseline & 87.84\% & 86.29\% & -1.55\% \\
        M4 Medium + Simple & 90.25\% & 84.31\% & \textbf{-5.94\%} \\
        M5 Medium + Additive & 88.34\% & 85.39\% & -2.95\% \\
        M6 Medium + Scaled DP & 88.17\% & 85.23\% & -2.94\% \\
        \bottomrule
    \end{tabular}
\end{table}

Dropout consistently degraded performance across all models. Attention-augmented models were particularly affected, with M4 suffering the largest drop of \qty{5.94}{\percent}. This suggests that the dataset of 2.2 million samples provides sufficient regularization through data diversity, and that attention mechanisms are sensitive to noise introduced by dropout in intermediate representations. All subsequent results use no dropout.
```

---

## 9. Section V: Attention Weight Analysis

### 9.1 Figure 5 erweitern (Optional - niedrigere Priorität)

- [ ] **Figure 5 um Additive und Scaled DP erweitern**

**Hinweis:** Erfordert Attention-Weight-Daten aus:
- `results/figures/M4_Medium_Simple_Attention/M4_Medium_Simple_Attention_attention_weights.npy`
- `results/figures/M5_Medium_Additive_Attention/M5_Medium_Additive_Attention_attention_weights.npy`
- `results/figures/M6_Medium_Scaled_DP_Attention/M6_Medium_Scaled_DP_Attention_attention_weights.npy`

**TODO:** Prüfen ob diese Dateien verfügbar sind, dann 3-Panel-Figure erstellen.

---

## 10. Section VI: Conclusion

### 10.1 Accuracy-Werte korrigieren

- [x] **Im ersten Bullet Point** ✅ (2026-01-26)

**Suchen:**
```latex
\item The LSTM model with Simple Attention achieves the highest prediction accuracy (\qty{89.71}{\percent}), outperforming the baseline LSTM by \qty{5}{\percent} and more complex attention mechanisms by approximately \qty{1}{\percent}.
```

**Ersetzen durch:**
```latex
\item The LSTM model with Simple Attention (M4) achieves the highest prediction accuracy (\qty{90.25}{\percent}), outperforming the medium baseline by \qty{2.41}{\percent} and more complex attention mechanisms by approximately \qty{2}{\percent}.
```

### 10.2 Trainingszeit-Aussage ersetzen

- [x] **Im zweiten Bullet Point** ✅ (2026-01-26)

**Suchen:**
```latex
\item Simpler model architectures achieve comparable performance with significantly reduced computational cost, with the simple model demonstrating 4$\times$ higher learning efficiency than the complex model.
```

**Ersetzen durch:**
```latex
\item All models achieve inference times below \qty{3}{\milli\second} on CPU, with Simple Attention adding only \qty{1.7}{\percent} latency overhead while providing the largest accuracy improvement, making it suitable for real-time deployment at over \qty{100}{\hertz}.
```

### 10.3 Future Work erweitern

- [x] **In der Future Work Liste** ✅ (2026-01-26)

**Suchen:**
```latex
Future work will explore: (1) larger window sizes with memory-efficient implementations,
```

**Ersetzen durch:**
```latex
Future work will explore: (1) the impact of sliding window size on the accuracy-latency trade-off for memory-constrained systems,
```

---

## 11. Sonstige Korrekturen

### 11.1 TODO-Kommentar im Text entfernen

- [x] **In Section I** ✅ (2026-01-26)

**Suchen und entfernen:**
```latex
% TODO: Replace reference - Xing et al. 2019 is about driver activity recognition, NOT steering angle prediction
```

### 11.2 Figure 3 Entscheidung

- [ ] **ENTSCHEIDUNG TREFFEN:** Figure 3 (Training Curves) behalten oder entfernen?

**Option A - Behalten:** 
- Zeigt Konvergenzverhalten
- Weniger prominent im Text erwähnen

**Option B - Entfernen:**
- Nicht zentral für neues Narrativ
- Spart Platz

**Empfehlung:** Behalten, aber Text kürzen.

---

## Checkliste für Review

Nach allen Änderungen prüfen:

- [x] Alle Accuracy-Werte konsistent (90.25% für M4, 87.84% für M3, etc.) ✅
- [x] Alle RMSE-Werte im neuen Format (~0.03-0.04, nicht ~0.001) ✅
- [x] Keine Referenzen auf TCR oder Learning Efficiency mehr ✅
- [x] Keine Referenzen auf Trainingszeit als Hauptmetrik ✅
- [x] Figure-Nummern nach Entfernung/Einfügen noch korrekt ✅
- [x] Table-Nummern nach Entfernung/Einfügen noch korrekt ✅
- [x] `\cref{}` Referenzen alle gültig ✅
- [x] Hardware: RTX 2060 Super (nicht 3070 Ti) ✅
- [x] Window Size nur in Future Work erwähnt ✅

---

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `/mnt/project/paper.tex` | Hauptdatei (READ-ONLY, kopieren nach /home/claude/) |
| `/mnt/project/model_evaluation_results.md` | Aktuelle Ergebnisse (Referenz) |
| `/mnt/project/model_evaluation_results_dropout.md` | Dropout-Ablation (Referenz) |

---

## Ausführungsreihenfolge

1. **Kopiere** `paper.tex` nach `/home/claude/paper.tex`
2. **Führe Änderungen durch** in der Reihenfolge dieser TODO-Liste
3. **Kompiliere** LaTeX zur Validierung: `pdflatex paper.tex`
4. **Prüfe** Referenzen und Nummerierungen
5. **Kopiere** fertiges Paper nach `/mnt/user-data/outputs/`

---

*Erstellt: 2026-01-26*
