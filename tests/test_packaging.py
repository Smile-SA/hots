"""
=====================
Testing rac packaging
=====================
"""
import subprocess
import rac


def test_rac_version(package_directory):
    """PEP 396 version available equals version from VERSION.txt"""
    file_version = (package_directory / "VERSION.txt").read_text()
    assert rac.__version__ == file_version.strip()


def test_rac_help_command():
    """Check a "rac" command is available"""
    result = subprocess.run(["rac", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Command available in $PATH
    assert result.returncode == 0

    # Command help displayed
    assert result.stdout.startswith(b"Usage: rac")
