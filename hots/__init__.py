"""Expose version from VERSION.txt"""

from pathlib import Path

try:
    __version__ = (
        Path(__file__).resolve().parent.parent
        / 'VERSION.txt'
    ).read_text().strip()
except Exception:
    __version__ = '0.0.0'
