from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from src.paths import CONFIG_DIR


load_dotenv()


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load a YAML file into a Python dictionary."""
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_configs() -> Dict[str, Dict[str, Any]]:
    """Load all project configuration files."""
    settings = load_yaml(CONFIG_DIR / "settings.yaml")
    models = load_yaml(CONFIG_DIR / "models.yaml")
    prompts = load_yaml(CONFIG_DIR / "prompts.yaml")

    return {
        "settings": settings,
        "models": models,
        "prompts": prompts,
    }