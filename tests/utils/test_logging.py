"""
Tests for utils.logging.
"""
import logging
from src.utils import logging as log_module

def test_logger_setup(caplog):
    logger = log_module.get_logger("test")
    with caplog.at_level(logging.INFO):
        logger.info("hello")
    assert "hello" in caplog.text
