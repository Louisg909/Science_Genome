"""
Configuration Loader.

Purpose:
    Load project settings (API keys, DB path, model options) from YAML/JSON.

Goals:
    - Keep sensitive data out of code.
    - Provide a single source of truth for parameters.

Inputs:
    - config.yaml or config.json file.

Outputs:
    - Python dict of configuration values.
"""
