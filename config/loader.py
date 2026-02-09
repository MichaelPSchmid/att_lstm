"""
Configuration loader for LSTM-Attention training.

Loads and merges base config with model-specific configs.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries. Override values take precedence.

    Args:
        base: Base dictionary
        override: Override dictionary (values take precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    model_config_path: str,
    base_config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load and merge configuration files.

    Args:
        model_config_path: Path to model-specific config (e.g., m1_small_baseline.yaml)
        base_config_path: Path to base config (default: config/base_config.yaml)

    Returns:
        Merged configuration dictionary
    """
    # Determine project root (parent of config/)
    project_root = Path(__file__).parent.parent.resolve()

    # Load base config
    if base_config_path is None:
        base_config_path = project_root / "config" / "base_config.yaml"
    else:
        base_config_path = Path(base_config_path)

    with open(base_config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Load model config
    model_config_path = Path(model_config_path)
    if not model_config_path.is_absolute():
        # Try relative to project root
        if not model_config_path.exists():
            model_config_path = project_root / model_config_path

    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    # Merge configs (model config overrides base config)
    config = deep_merge(base_config, model_config)

    return config


def get_model_class(model_type: str):
    """
    Get the model class based on type string.

    Args:
        model_type: One of "baseline", "simple_attention", "additive_attention",
                    "scaled_dp_attention", "mlp_last", "mlp_flat"

    Returns:
        Model class
    """
    if model_type == "baseline":
        from model.lstm_baseline import LSTMModel
        return LSTMModel
    elif model_type == "simple_attention":
        from model.lstm_simple_attention import LSTMAttentionModel
        return LSTMAttentionModel
    elif model_type == "additive_attention":
        from model.lstm_additive_attention import LSTMAttentionModel
        return LSTMAttentionModel
    elif model_type == "scaled_dp_attention":
        from model.lstm_scaled_dp_attention import LSTMScaleDotAttentionModel
        return LSTMScaleDotAttentionModel
    elif model_type in ("mlp_last", "mlp_flat"):
        from model.mlp_baseline import MLPModel
        return MLPModel
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_model_from_config(config: Dict[str, Any]):
    """
    Create a model instance from configuration.

    Args:
        config: Merged configuration dictionary

    Returns:
        Model instance
    """
    model_config = config["model"]
    training_config = config["training"]
    data_config = config.get("data", {})

    model_class = get_model_class(model_config["type"])
    model_type = model_config["type"]

    # MLP models have different parameters
    if model_type in ("mlp_last", "mlp_flat"):
        model_kwargs = {
            "input_size": model_config["input_size"],
            "hidden_sizes": model_config.get("hidden_sizes", [64, 64]),
            "output_size": model_config["output_size"],
            "lr": training_config["learning_rate"],
            "dropout": training_config.get("dropout", 0.0),
            "use_last_only": model_type == "mlp_last",
            "seq_len": data_config.get("window_size", 50),
        }
    else:
        # LSTM-based models
        model_kwargs = {
            "input_size": model_config["input_size"],
            "hidden_size": model_config["hidden_size"],
            "num_layers": model_config["num_layers"],
            "output_size": model_config["output_size"],
            "lr": training_config["learning_rate"],
            "dropout": training_config.get("dropout", 0.0),
        }

    return model_class(**model_kwargs)


def print_config(config: Dict[str, Any]) -> None:
    """Pretty print configuration."""
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("=" * 60)


if __name__ == "__main__":
    # Test loading
    import sys

    if len(sys.argv) > 1:
        config = load_config(sys.argv[1])
    else:
        config = load_config("config/model_configs/m1_small_baseline.yaml")

    print_config(config)
