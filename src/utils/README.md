# Utilities

Small helper functions shared across packages.

## Contents

- `config.py` – `load_config` reads YAML or JSON files into dictionaries.
- `logging.py` – `get_logger` provides a simple, project-wide logging setup.

## Usage

```python
from src.utils import config, logging

settings = config.load_config("settings.yml")
log = logging.get_logger(__name__)
log.info("Loaded %s sections", len(settings))
```

## Contributing

Keep utilities minimal and free from project-specific dependencies. New helpers
should be generic and include unit tests under `tests/utils`.
