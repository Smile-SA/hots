"""Testing hots packaging."""

import subprocess
from importlib.metadata import version
from importlib.resources import files

import hots


def test_hots_version(package_directory):
    """PEP 396 version available equals version from VERSION.txt."""
    file_version = (package_directory / "VERSION.txt").read_text()
    assert hots.__version__ == file_version.strip()


def test_hots_version_from_metadata():
    """The version comes from the installed distribution's metadata."""
    assert hots.__version__ == version("hots")


def test_hots_help_command():
    """Check a "hots" command is available."""
    result = subprocess.run(["hots", "--help"], capture_output=subprocess.PIPE)

    # Command available in $PATH
    assert result.returncode == 0

    # Command help displayed
    assert result.stdout.startswith(b"Usage: hots")


def test_hots_version_command():
    """`hots --version` prints the version without requiring --config."""
    result = subprocess.run(["hots", "--version"], capture_output=True, text=True)

    assert result.returncode == 0
    assert result.stdout.strip() == f"hots, version {hots.__version__}"


def test_sample_configs_are_package_data():
    """The sample configs the README points to ship with the package."""
    config_dir = files("hots.config")
    for name in ("sample_config_file.json", "sample_config_kafka.json"):
        assert config_dir.joinpath(name).is_file(), name
