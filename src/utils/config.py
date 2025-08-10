"""Configuration loader helpers.

Only a very small subset of features is implemented – enough for the tests and
for simple projects.  The function supports YAML and JSON files, returning the
contents as a ``dict``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a configuration file into a dictionary.

    Parameters
    ----------
    path:
        Path to a YAML or JSON configuration file.
    """

    p = Path(path)
    text = p.read_text(encoding="utf8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:  # pragma: no cover - defensive
            raise RuntimeError("PyYAML is required to read YAML files")
        return yaml.safe_load(text) or {}
    return json.loads(text)


__all__ = ["load_config"]

