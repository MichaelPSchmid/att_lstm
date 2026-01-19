# Training

## Übersicht

Das Training wird mit PyTorch Lightning durchgeführt, was standardisierte Trainingsloops, Logging und Checkpointing bietet.

---

## Haupt-Trainingsscript

**Datei:** `main.py`

### Konfiguration

```python
# Reproduzierbarkeit
pl.seed_everything(3407)

# GPU-Optimierung
torch.set_float32_matmul_precision('medium')

# Modell-Initialisierung
model = LSTMModel(
    input_size=5,
    hidden_size=128,
    num_layers=5,
    output_size=1,
    lr=0.000382819
)

# DataModule
data_module = TimeSeriesDataModule(
    feature_path="...",
    target_path="...",
    batch_size=32
)
```

### Trainer-Setup

```python
trainer = pl.Trainer(
    max_epochs=80,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=logger,
    enable_checkpointing=True,
    log_every_n_steps=50
)
```

---

## Callbacks

### Early Stopping

Stoppt das Training wenn sich `val_loss` nicht mehr verbessert:

```python
EarlyStopping(
    monitor="val_loss",
    patience=5,      # Anzahl Epochen ohne Verbesserung
    mode="min"       # Minimiere val_loss
)
```

### Model Checkpoint

Speichert die besten Modelle:

```python
ModelCheckpoint(
    monitor="val_loss",
    save_top_k=3,    # Behalte die 3 besten Modelle
    mode="min",
    filename="LSTMModel-{epoch:02d}-{val_loss:.4f}"
)
```

**Checkpoint-Pfad:** `lightning_logs/{ModelName}/version_X/checkpoints/`

---

## Logging

### TensorBoard Logger

```python
logger = TensorBoardLogger("lightning_logs", name="LSTMModel")
```

**Logs anzeigen:**
```bash
tensorboard --logdir lightning_logs
```

### Geloggte Metriken

| Phase | Metriken |
|-------|----------|
| Training | `train_loss` |
| Validation | `val_loss`, `val_rmse`, `val_mape`, `val_r2`, `avg_val_abs_accuracy` |
| Test | `test_loss`, `test_rmse`, `test_mape`, `test_r2`, `avg_test_abs_accuracy` |

---

## Hyperparameter-Optimierung mit Optuna

**Dateien:** `optuna/optuna1.py`, `optuna/optuna2.py`, `optuna/optuna3.py`

### Suchraum

```python
def objective(trial):
    hidden_size = trial.suggest_int("hidden_size", 32, 128, step=16)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    # ...
```

| Parameter | Bereich | Schrittweite |
|-----------|---------|--------------|
| `hidden_size` | 32 - 128 | 16 |
| `num_layers` | 1 - 5 | 1 |
| `lr` | 1e-5 - 1e-3 | log-uniform |

### Optuna-Konfiguration

```python
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
```

- **Optimierungsziel:** Minimiere `val_loss`
- **Anzahl Trials:** 50
- **Epochs pro Trial:** 10 (mit Early Stopping)

### Ausführung

```bash
python optuna/optuna1.py
```

---

## Training durchführen

### Neues Training starten

```python
# In main.py
trainer.fit(model, data_module)
```

### Von Checkpoint fortsetzen

```python
# Checkpoint laden
checkpoint_path = "lightning_logs/.../checkpoints/model.ckpt"
model = LSTMModel.load_from_checkpoint(checkpoint_path)

# Oder Training fortsetzen
trainer.fit(model, data_module, ckpt_path=checkpoint_path)
```

### Modell testen

```python
trainer.test(model, dataloaders=data_module.test_dataloader())
```

---

## Training-Parameter Übersicht

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| `max_epochs` | 80 | Maximale Epochen |
| `batch_size` | 32 | Samples pro Batch |
| `lr` | 0.000382819 | Learning Rate (optimiert) |
| `hidden_size` | 128 | LSTM Hidden Dimension |
| `num_layers` | 5 | LSTM Schichten |
| `early_stop_patience` | 5 | Epochen ohne Verbesserung |
| `seed` | 3407 | Random Seed |

---

## GPU-Optimierung

### Tensor Cores

```python
torch.set_float32_matmul_precision('medium')
```

Nutzt Tensor Cores für schnellere Matrix-Multiplikationen (leichter Genauigkeitsverlust).

### DataLoader

```python
DataLoader(
    ...,
    num_workers=15,    # Paralleles Laden
    pin_memory=True    # Schnellerer GPU-Transfer
)
```

---

## Vorhandene Trainingsläufe

Die `lightning_logs/` Verzeichnisstruktur:

```
lightning_logs/
├── LSTMModel/
│   ├── version_0/
│   ├── version_1/
│   └── ...
├── LSTMAttentionModel/
├── LSTMAdditiveAttentionModel/
├── LSTMScaledDotAttentionModel/
└── CNNModel/
```

Jede Version enthält:
- `hparams.yaml` - Hyperparameter
- `events.out.tfevents.*` - TensorBoard Logs
- `checkpoints/` - Gespeicherte Modelle

---

## Typischer Training-Workflow

1. **Daten vorbereiten**
   ```bash
   python preprocess/slice_window.py
   ```

2. **Hyperparameter optimieren** (optional)
   ```bash
   python optuna/optuna1.py
   ```

3. **Modell trainieren**
   ```bash
   python main.py
   ```

4. **Training überwachen**
   ```bash
   tensorboard --logdir lightning_logs
   ```

5. **Modell evaluieren**
   - Test-Metriken werden automatisch geloggt
   - Attention-Visualisierung mit `main_hot_map.py`
