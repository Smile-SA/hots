"""HOTS logging configuration."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(level: str, filename: Optional[str], fmt: str, add_console: bool = False) -> None:
    """
    Initialize Python root logger.
    :param level: e.g. "INFO" or "DEBUG"
    :param filename: path to log file, or None for stdout
    :param fmt: logging format string
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers = []
    formatter = logging.Formatter(fmt)

    if filename:
        p = Path(filename)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            p,
            maxBytes=5_000_000,
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

        if add_console:
            sh = logging.StreamHandler()
            sh.setLevel(numeric_level)
            sh.setFormatter(formatter)
            handlers.append(sh)
        
    else:
        sh = logging.StreamHandler()
        sh.setLevel(numeric_level)
        sh.setFormatter(formatter)
        handlers.append(sh)

    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True
    )
