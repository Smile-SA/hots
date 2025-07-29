"""HOTS logging configuration."""

import logging
from typing import Optional


def setup_logging(level: str, filename: Optional[str], fmt: str) -> None:
    """
    Initialize Python root logger.
    :param level: e.g. "INFO" or "DEBUG"
    :param filename: path to log file, or None for stdout
    :param fmt: logging format string
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers = []
    if filename:
        handlers.append(logging.FileHandler(filename))
    else:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        handlers=handlers,
    )
