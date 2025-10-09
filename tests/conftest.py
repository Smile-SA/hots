"""Fixtures for pytest."""

import os
import pathlib
import random
import tempfile
from datetime import datetime, timezone

import pytest


@pytest.fixture
def frozen_time(monkeypatch):
    """Fixture to freeze datetime.now() at a given moment for deterministic tests."""
    class _Frozen:
        def __init__(self, dt: datetime):
            self.dt = dt.replace(tzinfo=timezone.utc)

        def __enter__(self):
            import datetime as _dt

            class _Fixed(_dt.datetime):
                @classmethod
                def now(cls, tz=None):
                    return self.dt if tz is None else self.dt.astimezone(tz)
            monkeypatch.setattr(_dt, 'datetime', _Fixed)
            return self.dt

        def __exit__(self, *exc):
            return False
    return _Frozen


@pytest.fixture(scope='session')
def rng_seed():
    """Fixture providing a fixed random seed for reproducible tests."""
    return 12345


@pytest.fixture
def rng(rng_seed):
    """Fixture returning a random.Random instance seeded with rng_seed."""
    random.seed(rng_seed)
    return random.Random(rng_seed)


@pytest.fixture
def tmp_dir():
    """Fixture yielding a temporary directory that is cleaned up after use."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture(scope='session')
def test_settings():
    """Fixture exposing test configuration values like DB_URL and API_BASE."""
    return {
        'DB_URL': os.getenv('TEST_DB_URL', 'sqlite:///:memory:'),
        'API_BASE': os.getenv('TEST_API_BASE', 'http://localhost:8000'),
    }


@pytest.fixture
def tests_directory():
    """Get path of parent dir."""
    return pathlib.Path(__file__).resolve().parent


@pytest.fixture
def package_directory(tests_directory):
    """Get test parent dir."""
    return tests_directory.parent


@pytest.fixture
def tests_data_directory(tests_directory):
    """Check if data dir is in test dir."""
    return tests_directory / 'data'
