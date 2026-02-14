"""Configuration loader for GoalFlow."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, loads default.yaml

    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        # Try to find default.yaml in various locations
        possible_paths = [
            Path(__file__).parent.parent / "configs" / "default.yaml",
            Path("configs/default.yaml"),
            Path("./goalflow/configs/default.yaml"),
        ]
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path is None:
        raise FileNotFoundError("Could not find default.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set default values for missing keys
    config = set_defaults(config)

    return config


def set_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Set default values for missing configuration keys."""
    defaults = {
        "data": {
            "data_root": "./data",
            "augmentation": {"enabled": False},
        },
        "training": {
            "num_workers": 4,
            "pin_memory": True,
            "accelerator": "auto",
        },
    }

    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey not in config[key]:
                    config[key][subkey] = subvalue

    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
