"""Expose the package version."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _read_dist_version() -> str | None:
    """Version of the installed distribution (VERSION.txt at build time)."""
    try:
        return version("hots")
    except PackageNotFoundError:
        return None


def _read_version_txt() -> str | None:
    """VERSION.txt of a source checkout that isn't installed.

    Returns None if not found/readable.
    """
    # repo layout: <root>/hots/__init__.py and <root>/VERSION.txt
    p = Path(__file__).resolve().parent.parent / "VERSION.txt"
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return None


__version__ = _read_dist_version() or _read_version_txt() or "0.0.0"
