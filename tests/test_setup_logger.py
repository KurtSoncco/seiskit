"""Tests for seiskit.setup_logger module."""

import logging

from seiskit.setup_logger import setup_basic_logger


def test_setup_basic_logger_returns_logger():
    """setup_basic_logger returns a logger instance."""
    logger = setup_basic_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "seiskit.setup_logger"
