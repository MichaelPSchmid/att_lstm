"""
Model configurations - Single Source of Truth.

Contains all model definitions used across the evaluation scripts.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    id: str
    name: str
    type: str
    config_no_dropout: str
    config_dropout: str
    log_dir_no_dropout: str
    log_dir_dropout: str


# All available models - Single Source of Truth
MODELS: List[ModelConfig] = [
    ModelConfig(
        id="m1",
        name="M1 MLP Last",
        type="MLP (5→64→64→1)",
        config_no_dropout="config/model_configs/m1_mlp_last.yaml",
        config_dropout="config/model_configs/m1_mlp_last.yaml",  # No dropout variant
        log_dir_no_dropout="M1_MLP_Last",
        log_dir_dropout="M1_MLP_Last",
    ),
    ModelConfig(
        id="m2",
        name="M2 MLP Flat",
        type="MLP (250→128→64→1)",
        config_no_dropout="config/model_configs/m2_mlp_flat.yaml",
        config_dropout="config/model_configs/m2_mlp_flat.yaml",  # No dropout variant
        log_dir_no_dropout="M2_MLP_Flat",
        log_dir_dropout="M2_MLP_Flat",
    ),
    ModelConfig(
        id="m3",
        name="M3 Small Baseline",
        type="LSTM (64, 3)",
        config_no_dropout="config/model_configs/m3_small_baseline.yaml",
        config_dropout="config/model_configs/m3_small_baseline_dropout.yaml",
        log_dir_no_dropout="M3_Small_Baseline",
        log_dir_dropout="M3_Small_Baseline_Dropout",
    ),
    ModelConfig(
        id="m4",
        name="M4 Small + Simple Attn",
        type="LSTM + Attention (64, 3)",
        config_no_dropout="config/model_configs/m4_small_simple_attn.yaml",
        config_dropout="config/model_configs/m4_small_simple_attn_dropout.yaml",
        log_dir_no_dropout="M4_Small_Simple_Attention",
        log_dir_dropout="M4_Small_Simple_Attention_Dropout",
    ),
    ModelConfig(
        id="m5",
        name="M5 Medium Baseline",
        type="LSTM (128, 5)",
        config_no_dropout="config/model_configs/m5_medium_baseline.yaml",
        config_dropout="config/model_configs/m5_medium_baseline_dropout.yaml",
        log_dir_no_dropout="M5_Medium_Baseline",
        log_dir_dropout="M5_Medium_Baseline_Dropout",
    ),
    ModelConfig(
        id="m6",
        name="M6 Medium + Simple Attn",
        type="LSTM + Attention (128, 5)",
        config_no_dropout="config/model_configs/m6_medium_simple_attn.yaml",
        config_dropout="config/model_configs/m6_medium_simple_attn_dropout.yaml",
        log_dir_no_dropout="M6_Medium_Simple_Attention",
        log_dir_dropout="M6_Medium_Simple_Attention_Dropout",
    ),
    ModelConfig(
        id="m7",
        name="M7 Medium + Additive Attn",
        type="LSTM + Additive (128, 5)",
        config_no_dropout="config/model_configs/m7_medium_additive_attn.yaml",
        config_dropout="config/model_configs/m7_medium_additive_attn_dropout.yaml",
        log_dir_no_dropout="M7_Medium_Additive_Attention",
        log_dir_dropout="M7_Medium_Additive_Attention_Dropout",
    ),
    ModelConfig(
        id="m8",
        name="M8 Medium + Scaled DP",
        type="LSTM + Scaled DP (128, 5)",
        config_no_dropout="config/model_configs/m8_medium_scaled_dp_attn.yaml",
        config_dropout="config/model_configs/m8_medium_scaled_dp_attn_dropout.yaml",
        log_dir_no_dropout="M8_Medium_Scaled_DP_Attention",
        log_dir_dropout="M8_Medium_Scaled_DP_Attention_Dropout",
    ),
]

# Quick access by model ID
MODEL_BY_ID: Dict[str, ModelConfig] = {m.id: m for m in MODELS}
