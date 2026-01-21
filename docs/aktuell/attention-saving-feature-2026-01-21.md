# Feature: Attention Weight Saving & Visualization - 2026-01-21

## Status
- **Branch:** `feature/attention-saving`
- **Status:** Fertig, bereit zum Merge

## Übersicht

Dieses Feature ermöglicht das automatische Speichern und Visualisieren von Attention Weights während Training und Evaluation.

## Neue Funktionalität

### 1. Attention Weights während Training speichern

**Config-basiert:** In den Attention-Model-Configs ist `attention.enabled: true` gesetzt.

```yaml
# config/model_configs/m5_medium_additive_attn.yaml
attention:
  enabled: true  # Automatisch aktiviert für Attention-Modelle
```

**CLI-Override:**
```bash
# Erzwingen (auch wenn Config false sagt)
python scripts/train_model.py --config ... --save-attention

# Deaktivieren (auch wenn Config true sagt)
python scripts/train_model.py --config ... --no-save-attention

# Eigenes Ausgabeverzeichnis
python scripts/train_model.py --config ... --attention-dir results/my_attention/
```

**Output:**
```
attention_weights/<model_name>/
├── attention_epoch_000.npy    # Pro Validation-Epoch
├── attention_epoch_001.npy
├── ...
├── attention_test.npy         # Test-Set (am Ende)
├── attention_all_epochs.npz   # Alle Daten kombiniert
└── attention_all_epochs.csv   # Für einfache Analyse
```

### 2. Attention Heatmap bei Evaluation

Bei der Evaluation wird automatisch eine Attention-Heatmap generiert (für Attention-Modelle).

```bash
python scripts/evaluate_model.py \
    --checkpoint lightning_logs/M5_.../best.ckpt \
    --config config/model_configs/m5_medium_additive_attn.yaml
```

**Output:**
```
results/figures/<model_name>/
├── <model_name>_scatter.png           # Predicted vs Actual
├── <model_name>_residuals.png         # Residual Distribution
├── <model_name>_timeline.png          # Prediction Timeline
├── <model_name>_error_distribution.png
├── <model_name>_attention_heatmap.png # NEU: Attention Visualization
└── <model_name>_attention_weights.npy # NEU: Raw Attention Data
```

**Visualisierung:**
- **Matrix-Attention (Additive):** Heatmap der (seq_len x seq_len) Attention-Matrix
- **1D-Attention (Simple, ScaledDP):** Bar-Plot der Attention-Weights pro Timestep

### 3. Predictions als CSV speichern

```bash
python scripts/evaluate_model.py \
    --checkpoint ... \
    --config ... \
    --save-predictions
```

**Output:** `results/predictions/<model_name>_predictions.csv`
```csv
sample_idx,prediction,target,error,abs_error
0,0.123,0.125,-0.002,0.002
1,0.456,0.450,0.006,0.006
...
```

## Geänderte Dateien

### Modelle (return_attention Parameter)
- `model/LSTM_attention.py`
- `model/diff_attention/additive_attention.py`
- `model/diff_attention/scaled_dot_product.py`

### Neuer Callback
- `scripts/callbacks/__init__.py`
- `scripts/callbacks/attention_callback.py`

### Erweiterte Skripte
- `scripts/train_model.py` - `--save-attention`, `--no-save-attention`, `--attention-dir`
- `scripts/evaluate_model.py` - Attention Heatmap, `--save-predictions`, `--no-attention-plot`

### Config
- `config/base_config.yaml` - `attention.enabled`, `attention.save_per_epoch`, etc.
- `config/model_configs/m2_*.yaml` - `attention.enabled: true`
- `config/model_configs/m4_*.yaml` - `attention.enabled: true`
- `config/model_configs/m5_*.yaml` - `attention.enabled: true`
- `config/model_configs/m6_*.yaml` - `attention.enabled: true`

## Gelöschte Dateien (Redundant)

| Alte Datei | Ersetzt durch |
|------------|---------------|
| `main.py` | `scripts/train_model.py` |
| `main_hot_map.py` | `scripts/train_model.py` + Attention Callback |
| `main_hot_map2.py` | `scripts/train_model.py` + Attention Callback |
| `eval_hot_map_2.py` | `scripts/evaluate_model.py` (Attention Heatmap) |
| `plot_prediction_target.py` | `scripts/evaluate_model.py` (Timeline Plot) |
| `save_test_results.py` | `scripts/evaluate_model.py --save-predictions` |
| `test_predict_actual.py` | Nicht mehr benötigt |
| `save_sequenceID_timestep.py` | Nicht mehr benötigt |
| `model/attention_visualization/` | In Hauptmodelle integriert |

## Commits

```
8386bab feat(eval): Add attention heatmap and predictions export to evaluate_model.py
7f8029b feat(config): Enable attention saving via config instead of CLI flag
4299254 refactor: Remove redundant attention visualization scripts
f0fe69d feat(attention): Add attention weight saving during training
```

## Beispiel-Workflow

```bash
# 1. Training mit automatischem Attention-Saving (aus Config)
python scripts/train_model.py \
    --config config/model_configs/m5_medium_additive_attn.yaml

# 2. Evaluation mit allen Visualisierungen
python scripts/evaluate_model.py \
    --checkpoint lightning_logs/M5_Medium_Additive_Attention/version_0/checkpoints/best.ckpt \
    --config config/model_configs/m5_medium_additive_attn.yaml \
    --save-predictions \
    --output results/m5_results.json

# 3. Attention Weights analysieren (Python)
import numpy as np
weights = np.load("attention_weights/M5_Medium_Additive_Attention/attention_all_epochs.npz")
print(weights.files)  # ['epoch_0', 'epoch_1', ..., 'test', 'time_steps']
```

## Nächste Schritte

1. [ ] Branch `feature/attention-saving` in `main` mergen
2. [ ] M4-M6 Training abwarten
3. [ ] Evaluation mit neuen Features durchführen
4. [ ] Attention-Analyse für Paper dokumentieren
