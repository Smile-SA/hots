"""HOTS logging configuration."""

import logging


def setup_logging(level=logging.INFO):
    """Configure the root logger with a standard format and level."""
    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        level=level,
    )
