"""
===================
Fixtures for pytest
===================
"""

import pathlib

import pytest


@pytest.fixture
def tests_directory():
    return pathlib.Path(__file__).resolve().parent


@pytest.fixture
def package_directory(tests_directory):
    return tests_directory.parent


@pytest.fixture
def tests_data_directory(tests_directory):
    return tests_directory / "data"
