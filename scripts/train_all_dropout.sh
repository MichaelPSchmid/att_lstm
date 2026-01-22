#!/bin/bash
# Train all models WITH dropout sequentially
# Usage: bash scripts/train_all_dropout.sh

echo "=========================================="
echo "Starting Ablation Study - With Dropout"
echo "=========================================="
echo ""

CONFIGS=(
    "config/model_configs/m1_small_baseline_dropout.yaml"
    "config/model_configs/m2_small_simple_attn_dropout.yaml"
    "config/model_configs/m3_medium_baseline_dropout.yaml"
    "config/model_configs/m4_medium_simple_attn_dropout.yaml"
    "config/model_configs/m5_medium_additive_attn_dropout.yaml"
    "config/model_configs/m6_medium_scaled_dp_attn_dropout.yaml"
)

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training: $config"
    echo "Started: $(date)"
    echo "=========================================="

    python scripts/train_model.py --config "$config"

    echo ""
    echo "Finished: $(date)"
    echo ""
done

echo "=========================================="
echo "All training completed!"
echo "Finished: $(date)"
echo "=========================================="
