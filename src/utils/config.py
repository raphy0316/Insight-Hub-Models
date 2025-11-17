from __future__ import annotations

from typing import Any, Dict
import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


