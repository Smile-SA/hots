"""HOTS logging configuration."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(
        level: str, filename: Optional[str], fmt: str,
        add_console: bool = False, reset_file: bool = True
) -> None:
    """Initialize Python root logger with a fresh file per run when reset_file=True."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)

    handlers = []

    if filename:
        p = Path(filename)
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)

        if reset_file and p.exists():
            try:
                p.unlink()
            except Exception:
                pass

        fh = RotatingFileHandler(
            p, mode='w', maxBytes=5_000_000, backupCount=3, encoding='utf-8', delay=False
        )
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        handlers.append(fh)

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

    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)
