from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from src.paths import CONFIG_DIR


load_dotenv()


ConfigDict = Dict[str, Any]


def load_yaml(file_path: Path) -> ConfigDict:
    """Load a YAML file into a Python dictionary."""
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {file_path}")
    return data


@lru_cache(maxsize=1)
def load_all_configs() -> Dict[str, ConfigDict]:
    """Load and cache all configuration files."""
    return {
        'settings': load_yaml(CONFIG_DIR / 'settings.yaml'),
        'models': load_yaml(CONFIG_DIR / 'models.yaml'),
        'prompts': load_yaml(CONFIG_DIR / 'prompts.yaml'),
    }


def get_nested(config: ConfigDict, dotted_key: str, default: Optional[Any] = None) -> Any:
    """Safely get a nested value using dotted keys."""
    current: Any = config
    for key in dotted_key.split('.'):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def get_config_value(section: str, *keys: str, default: Optional[Any] = None) -> Any:
    """Convenience accessor for nested configuration values."""
    configs = load_all_configs()
    current: Any = configs.get(section, {})
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
