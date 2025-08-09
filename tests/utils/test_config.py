"""
Tests for utils.config.
"""
from src.utils import config
import tempfile, yaml

def test_load_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    sample = {"db_path": "science_papers.db"}
    cfg_path.write_text(yaml.dump(sample))
    cfg = config.load_config(cfg_path)
    assert cfg["db_path"] == "science_papers.db"
