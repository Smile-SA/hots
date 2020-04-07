.. _installation:

============
Installation
============

Requirements
============

:term:`rac` works on any platform with Python 3.6 and up, a graphic display on any OS (\*nix or
Windows)?

Production
==========

.. note::

   This operating mode is intended for end users.

.. code:: console

   pip install rac

.. hint::

   If this raises a security error, please prefix the command with ``sudo ...`` or login as "root"
   or system administrator.

.. todo:: Docker image installation

Development
===========

.. note::

   This operating mode is intended for application maintainers.

You should create and activate a dedicated Python virtual environment with whatever tool you prefer
(virtualenv, venv, pew, pyenv, ...). The instruction that follow assume this virtual environment is
activated unless you may have issues.

Clone the source code from the |vcs_server|, then:

.. code:: console

   cd /where/you/cloned/rac
   pip install -e .[dev]

This last command installs the :command:`rac` command and the :term:`rac` package in the virtual
environment in "editable" mode. This means that every change you make in the Python source code is
immediately active (you don't need to reinstall after each change).

.. hint::

   If you need to change this documentation too, you may rather replace the last command with:

   .. code:: console

      pip installe -e .[dev,doc]
