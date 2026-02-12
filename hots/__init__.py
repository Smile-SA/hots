"""Expose version from VERSION.txt"""

from __future__ import annotations

from pathlib import Path


def _read_version_txt() -> str | None:
    """Best-effort read of VERSION.txt.

    Returns None if not found/readable.
    """
    candidates = [
        # repo layout: <root>/hots/__init__.py and <root>/VERSION.txt
        Path(__file__).resolve().parent.parent / "VERSION.txt",
        # installed package with data-files may place VERSION.txt next to package
        Path(__file__).resolve().parent / "VERSION.txt",
    ]
    for p in candidates:
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return None


def _read_dist_version() -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return None

    try:
        return version("hots")
    except PackageNotFoundError:
        return None


__version__ = _read_version_txt() or _read_dist_version() or "0.0.0"
