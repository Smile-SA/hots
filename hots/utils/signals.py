"""HOTS signal handling utilities."""

import signal


def setup_signal_handlers(shutdown_fn):
    """Register SIGINT and SIGTERM to the given shutdown function."""
    signal.signal(signal.SIGINT, shutdown_fn)
    signal.signal(signal.SIGTERM, shutdown_fn)
