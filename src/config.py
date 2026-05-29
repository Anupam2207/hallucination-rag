from __future__ import annotations

import os
from copy import deepcopy
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

    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {file_path}")
    return data


def deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Return ``base`` recursively merged with ``override`` without mutating either."""
    merged: ConfigDict = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _requested_profile(base_settings: ConfigDict) -> str | None:
    """Resolve the optional configuration profile name.

    Priority:
    1. ``HALLUCINATION_RAG_PROFILE`` environment variable.
    2. ``runtime.profile`` in ``configs/settings.yaml``.

    ``demo`` is the project default.  The base settings are already demo-safe, so
    loading ``settings_demo.yaml`` mainly documents and validates the profile.
    """
    profile = os.getenv("HALLUCINATION_RAG_PROFILE", "").strip().lower()
    if profile:
        return profile
    runtime = base_settings.get("runtime", {}) if isinstance(base_settings, dict) else {}
    configured = str(runtime.get("profile", "") or "").strip().lower()
    return configured or None


def load_settings_config(profile: str | None = None) -> ConfigDict:
    """Load settings.yaml and merge an optional settings_<profile>.yaml overlay."""
    settings = load_yaml(CONFIG_DIR / "settings.yaml")
    resolved_profile = (profile or _requested_profile(settings) or "").strip().lower()
    if resolved_profile:
        profile_path = CONFIG_DIR / f"settings_{resolved_profile}.yaml"
        if profile_path.exists():
            settings = deep_merge(settings, load_yaml(profile_path))
        elif profile not in (None, ""):
            raise FileNotFoundError(f"Config profile '{resolved_profile}' not found: {profile_path}")
    return settings


@lru_cache(maxsize=4)
def load_all_configs(profile: str | None = None) -> Dict[str, ConfigDict]:
    """Load and cache all configuration files.

    The optional profile argument supports low-resource demo mode and full
    research mode without changing public configuration accessors.
    """
    return {
        "settings": load_settings_config(profile),
        "models": load_yaml(CONFIG_DIR / "models.yaml"),
        "prompts": load_yaml(CONFIG_DIR / "prompts.yaml"),
    }


def get_nested(config: ConfigDict, dotted_key: str, default: Optional[Any] = None) -> Any:
    """Safely get a nested value using dotted keys."""
    current: Any = config
    for key in dotted_key.split("."):
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


def get_first_config_value(*paths: tuple[str, ...], default: Optional[Any] = None) -> Any:
    """Return the first existing config value from a list of section/key paths.

    Example:
        get_first_config_value(("settings", "detection", "support_threshold"),
                               ("settings", "detection", "similarity_support_threshold"),
                               default=0.70)
    """
    for path in paths:
        if not path:
            continue
        section, *keys = path
        value = get_config_value(section, *keys, default=None)
        if value is not None:
            return value
    return default
