# Refactoring: Code an Implementierungsplan anpassen - 2026-01-20

## Status
- **M1 (Small Baseline) Training läuft!**
- Branch: `refactor/preprocessing-efficiency`
- Nächster Schritt: Evaluation-Skripte erstellen

## Erledigte Aufgaben heute

### Preprocessing
- [x] `preprocess_parallel.py` mit `--max-files` Parameter
- [x] Separate Ordner: `prepared_dataset/` (paper) vs `prepared_dataset_full/` (full)
- [x] Paper-Dataset (5001 files) und Full-Dataset (21000 files) existieren

### Config-System ✅
- [x] `config/base_config.yaml` - gemeinsame Einstellungen
- [x] `config/model_configs/m1-m6_*.yaml` - 6 Modell-Configs
- [x] `config_loader.py` - lädt und merged Configs
- [x] `scripts/train_model.py` - CLI mit `--config`, `--dry-run`, `--resume`

### Bugfixes
- [x] `config.py`: `feature_` → `features_`, `target_` → `targets_`
- [x] `data_module.py`: Target Shape `[N]` → `[N, 1]` (Match mit Model Output)
- [x] `data_module.py`: `persistent_workers=True` für DataLoader
- [x] `pyproject.toml`: `tensorboard` und `pyyaml` hinzugefügt
- [x] Seed: 3407 → 42

## Offene Aufgaben

### Phase 2: Evaluation (nächste Session)
- [ ] `scripts/evaluate_model.py` - Test-Set Evaluation mit Metriken
- [ ] `scripts/compare_results.py` - Vergleichstabellen generieren
- [ ] CPU Inference-Messung mit Warm-up

### Training (läuft/geplant)
- [x] M1 Small Baseline - **läuft gerade** (~2h)
- [ ] M2 Small + Simple Attention
- [ ] M3 Small + Additive Attention
- [ ] M4 Small + Scaled Dot-Product
- [ ] M5 Medium Baseline
- [ ] M6 Large Baseline

## Wichtige Pfade

| Was | Pfad |
|-----|------|
| Configs | `config/base_config.yaml`, `config/model_configs/` |
| Training-Skript | `scripts/train_model.py` |
| Checkpoints | `lightning_logs/{model_name}/version_X/checkpoints/` |
| Paper-Dataset | `data/prepared_dataset/` (5001 files, 4.2GB) |
| Full-Dataset | `data/prepared_dataset_full/` (21000 files, 18GB) |

## Kommandos

```bash
# Training starten
python scripts/train_model.py --config config/model_configs/m1_small_baseline.yaml

# Dry-run (nur Config anzeigen)
python scripts/train_model.py --config config/model_configs/m2_small_simple_attn.yaml --dry-run

# TensorBoard
tensorboard --logdir lightning_logs/
```

## Commits auf diesem Branch
```
4ac8530 feat(training): Add YAML config system and CLI training script
e9016e7 feat(config): Support separate directories for paper and full datasets
cc6e781 feat(preprocess): Add --max-files parameter and clarify dataset size in paper
bc39ea5 refactor(preprocess): Add parallel preprocessing with memory-efficient numpy output
```

## Nächste Session
1. Prüfen ob M1 Training erfolgreich war
2. `scripts/evaluate_model.py` erstellen
3. M2 Training starten
4. Bugfixes committen
