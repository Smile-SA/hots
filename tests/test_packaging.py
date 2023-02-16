"""
=====================
Testing hots packaging
=====================
"""
import subprocess

import hots


def test_hots_version(package_directory):
    """PEP 396 version available equals version from VERSION.txt"""
    file_version = (package_directory / 'VERSION.txt').read_text()
    assert hots.__version__ == file_version.strip()


def test_hots_help_command():
    """Check a "hots" command is available"""
    result = subprocess.run(['hots', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Command available in $PATH
    assert result.returncode == 0

    # Command help displayed
    assert result.stdout.startswith(b'Usage: hots')
