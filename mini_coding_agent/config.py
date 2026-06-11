"""Configuration loading for Mini-Coding-Agent."""

from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Falls back to the bundled ``config/default.yaml`` if no path is given.
    """
    import yaml

    target = Path(path) if path else DEFAULT_CONFIG_PATH
    with target.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
