"""
=====================
Testing cots packaging
=====================
"""
import subprocess

import cots


def test_cots_version(package_directory):
    """PEP 396 version available equals version from VERSION.txt"""
    file_version = (package_directory / 'VERSION.txt').read_text()
    assert cots.__version__ == file_version.strip()


def test_cots_help_command():
    """Check a "cots" command is available"""
    result = subprocess.run(['cots', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Command available in $PATH
    assert result.returncode == 0

    # Command help displayed
    assert result.stdout.startswith(b'Usage: cots')
