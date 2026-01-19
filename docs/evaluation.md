# Evaluation

## Metriken

### 1. MSE / Loss

**Mean Squared Error** - Hauptmetrik für Training und Optimierung.

```python
loss = torch.mean((outputs - Y_batch) ** 2)
```

### 2. RMSE

**Root Mean Squared Error** - Interpretierbare Fehlermetrik in der gleichen Einheit wie das Target.

```python
rmse = torch.sqrt(torch.mean((outputs - Y_batch) ** 2))
```

**Interpretation:** Ein RMSE von 0.05 bedeutet, dass die Vorhersagen im Schnitt um 0.05 (normalisierte Torque-Einheiten) vom tatsächlichen Wert abweichen.

### 3. MAPE

**Mean Absolute Percentage Error** - Relativer Fehler.

```python
mape = torch.mean(torch.abs((outputs - Y_batch) / (Y_batch + 1e-8)))
```

**Hinweis:** `1e-8` wird addiert um Division durch Null zu vermeiden.

**Vorsicht:** MAPE kann bei Werten nahe 0 sehr groß werden, da `steerFiltered` Werte nahe 0 haben kann.

### 4. R² Score

**Bestimmtheitsmaß** - Anteil der erklärten Varianz.

```python
ss_res = torch.sum((outputs - Y_batch) ** 2)
ss_tot = torch.sum((Y_batch - torch.mean(Y_batch)) ** 2)
r2_score = 1 - ss_res / (ss_tot + 1e-8)
```

**Interpretation:**
- R² = 1.0: Perfekte Vorhersage
- R² = 0.0: Modell so gut wie Mittelwert
- R² < 0.0: Modell schlechter als Mittelwert

### 5. Absolute Accuracy

**Anteil der Vorhersagen innerhalb einer Toleranz.**

```python
abs_threshold = 0.05
abs_correct = torch.abs(outputs - Y_batch) < abs_threshold
accuracy = abs_correct.sum() / Y_batch.numel()
```

**Interpretation:** Prozentsatz der Vorhersagen, die weniger als ±0.05 vom tatsächlichen Wert abweichen.

---

## Metriken-Übersicht

| Metrik | Bereich | Optimal | Verwendung |
|--------|---------|---------|------------|
| Loss (MSE) | [0, ∞) | 0 | Training, Early Stopping |
| RMSE | [0, ∞) | 0 | Fehlerinterpretation |
| MAPE | [0, ∞) | 0 | Relativer Fehler |
| R² | (-∞, 1] | 1 | Modellgüte |
| Abs. Accuracy | [0, 1] | 1 | Praktische Genauigkeit |

---

## Evaluation-Phasen

### Validation (während Training)

- Nach jeder Epoche
- Steuert Early Stopping
- Auswahl des besten Checkpoints

```python
def on_validation_epoch_end(self):
    avg_abs_accuracy = self.sum_abs_correct / self.total_samples
    avg_rmse = self.rmse_sum / self.total_samples
    # ...
    self.log("avg_val_abs_accuracy", avg_abs_accuracy)
```

### Test (nach Training)

- Einmalig auf ungesehenem Test-Set
- Finale Modellbewertung

```python
trainer.test(model, dataloaders=data_module.test_dataloader())
```

---

## Ergebnisse speichern

### Vorhersagen exportieren

**Datei:** `save_test_results.py` (auskommentierter Code in `main.py`)

```python
# Vorhersagen und Targets speichern
predictions_path = "evaluation/test_predictions.pkl"
targets_path = "evaluation/test_targets.pkl"

with torch.no_grad():
    for batch in test_dataloader:
        X_batch, Y_batch = batch
        Y_pred = model(X_batch)
        predictions.extend(Y_pred.cpu().numpy())
        targets.extend(Y_batch.cpu().numpy())

pickle.dump(predictions, open(predictions_path, 'wb'))
pickle.dump(targets, open(targets_path, 'wb'))
```

### Vorhandene Ergebnisse

```
evaluation/
├── LSTM_Attention_win15_64_3/
├── LSTM_win15_64_3/
├── test_predictions.pkl
└── test_targets.pkl
```

---

## Attention Visualisierung

### Heatmaps

**Dateien:** `main_hot_map.py`, `main_hot_map2.py`, `eval_hot_map_2.py`

Visualisieren, welche Zeitschritte das Modell für die Vorhersage wichtig findet.

### Vorhandene Visualisierungen

```
attention_visualization/
├── AdditiveAttention/
├── ScaledDotProductAttention/
└── matrix/
```

---

## Modellvergleich

### Vergleichs-Setup

**Datei:** `comparison/data_seed42.py`

Stellt sicher, dass alle Modelle mit den gleichen Daten-Splits verglichen werden.

### Vergleichsdimensionen

1. **Vorhersagegenauigkeit**
   - RMSE, R², Absolute Accuracy auf Test-Set

2. **Modellkomplexität**
   - Anzahl Parameter
   - Speicherbedarf

3. **Trainingszeit**
   - Zeit pro Epoche
   - Konvergenzgeschwindigkeit

4. **Interpretierbarkeit**
   - Attention Weights analysierbar?

---

## Interpretation der Ergebnisse

### Gute Werte (Richtwerte)

| Metrik | Gut | Sehr gut |
|--------|-----|----------|
| RMSE | < 0.1 | < 0.05 |
| R² | > 0.8 | > 0.95 |
| Abs. Accuracy (±0.05) | > 70% | > 90% |

### Typische Probleme

1. **Hoher MAPE bei niedrigen Targets**
   - `steerFiltered` nahe 0 → MAPE explodiert
   - Lösung: MAPE mit Vorsicht interpretieren

2. **R² negativ**
   - Modell schlechter als Mittelwert
   - Indikator für Underfitting oder falsche Daten

3. **Validation besser als Training**
   - Möglicher Bug in der Metrik-Berechnung
   - Oder: Data Leakage

---

## Reproduzierbarkeit

### Seed setzen

```python
pl.seed_everything(3407)
```

Setzt Seeds für:
- Python `random`
- NumPy
- PyTorch
- CUDA

### Deterministische Operationen

Für vollständige Reproduzierbarkeit auf GPU:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Weiterführende Analysen

### Fehleranalyse

1. **Wo macht das Modell Fehler?**
   - Bei hohen/niedrigen Torque-Werten?
   - Bei bestimmten Fahrsituationen?

2. **Zeitliche Analyse**
   - Fehler über die Sequenz
   - Attention Weights vs. Feature-Wichtigkeit

### Ablation Studies

1. **Einfluss der Window Size**
   - 15 vs. 50 Zeitschritte

2. **Einfluss einzelner Features**
   - Welches Feature trägt am meisten bei?

3. **Attention vs. keine Attention**
   - Quantitativer Vergleich
