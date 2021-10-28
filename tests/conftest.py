"""
===================
Fixtures for pytest
===================
"""

import pathlib

import pytest


@pytest.fixture
def tests_directory():
    """Get path of parent dir."""
    return pathlib.Path(__file__).resolve().parent


@pytest.fixture
def package_directory(tests_directory):
    """Get test paretn dir"""
    return tests_directory.parent


@pytest.fixture
def tests_data_directory(tests_directory):
    """Check if data dir is in test dir."""
    return tests_directory / 'data'
