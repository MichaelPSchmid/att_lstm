# Refactoring: Code an Implementierungsplan anpassen - 2026-01-20

## Status
- Aktiver Schritt: Analyse und Planung
- Nächster: AP1 - Config-System implementieren

## Kontext
- Der aktuelle Code funktioniert, ist aber weniger strukturiert als der Implementierungsplan
- Ziel: Code an `docs/implementation_plan_lstm_attention.md` anpassen
- Branch: `refactor/preprocessing-efficiency`

## Hauptunterschiede Plan vs. Code

| Aspekt | Plan | Aktuell | Priorität |
|--------|------|---------|-----------|
| Config-System | YAML pro Modell | Hardcoded | Hoch |
| Seed | 42 | 3407 | Mittel |
| Inference-Messung | CPU + Warm-up | Fehlt | Mittel |
| Projektstruktur | `src/` Hierarchie | Flache Struktur | Niedrig |
| Features | 5 Features (andere Namen) | 5 Features (passt) | - |

## Offene Aufgaben

### Phase 1: Config-System ✅
- [x] `config/base_config.yaml` erstellen
- [x] `config/model_configs/` für M1-M6 erstellen
- [x] Config-Loader implementieren (`config_loader.py`)
- [x] `scripts/train_model.py` mit CLI erstellen

### Phase 2: Vereinheitlichung
- [x] Seed auf 42 ändern (in `main.py` und `base_config.yaml`)
- [ ] Modelle mit einheitlicher Schnittstelle
- [ ] Inference-Messung auf CPU implementieren

### Phase 3: Evaluation
- [ ] `scripts/evaluate_model.py` erstellen
- [ ] `scripts/compare_results.py` erstellen
- [ ] Warm-up für Inferenz-Messung

## Entscheidungen/Erkenntnisse
- [2026-01-20] Features im Plan (`speed`, `angle`, etc.) entsprechen den aktuellen Features (`vEgo`, `steeringAngleDeg`, etc.) - nur andere Namen
- [2026-01-20] Projektstruktur-Umbau (→ `src/`) hat niedrige Priorität, da funktional kein Unterschied
- [2026-01-20] Fokus auf Config-System und Training-Skript
- [2026-01-20] Phase 1 abgeschlossen: Config-System implementiert mit YAML-Dateien und CLI-Skript

## Nächste Session
- Mit Config-System beginnen (`base_config.yaml`)
- Dann `train_model.py` mit `--config` Parameter

## Referenzen
- Implementierungsplan: `docs/implementation_plan_lstm_attention.md`
- Aktueller Code: `model/`, `main.py`, `data_module.py`
- Paper: `docs/paper/paper.tex`
