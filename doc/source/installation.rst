.. _installation:

============
Installation
============

Requirements
============

:term:`hots` works on any platform with Python 3.8 and up, a graphic display on any \*nix-like OS.

Due to some needed packages installation, the `dev` Python version is required. Make sure to install
the `dev` version of Python you will use. For example, if you want to use Python 3.10, you can
install the package `python3.10-dev`.

:term:`hots` uses a solver in order to solve some optimization problems, through the use of `Pyomo`
(see :ref:`pyomo` for more information about its use). The user can use any solver working with
`Pyomo`, but this solver needs to be installed and indicated to `Pyomo`. By default, :term:`hots`
uses `GLPK`, an open-source solver which is installed with :term:`hots`. But to use `GLPK`, the 
user needs to install the following packages (through `apt` for example) :

* libglpk-dev
* glpk-utils

Production
==========

.. note::

   This operating mode is intended for end users.

.. code:: console

   pip install hots

.. hint::

   If this raises a security error, please prefix the command with ``sudo ...`` or login as "root"
   or system administrator.

.. todo:: Docker image installation

Development
===========

.. note::

   This operating mode is intended for application maintainers.

.. note::

   You can find a `Makefile` that creates a virtual environment and install all the needed packages
   and hots. If you do that, do not forget to activate the virtual environment before running `hots`.

You should create and activate a dedicated Python virtual environment with whatever tool you prefer
(virtualenv, venv, pew, pyenv, ...). The instruction that follow assume this virtual environment is
activated unless you may have issues.

Clone the source code from the |vcs_server|, then:

.. code:: console

   cd /where/you/cloned/hots
   pip install -e .[dev]

This last command installs the :command:`hots` command and the :term:`hots` package in the virtual
environment in "editable" mode. This means that every change you make in the Python source code is
immediately active (you don't need to reinstall after each change).

.. hint::

   If you need to change this documentation too, you may rather replace the last command with:

   .. code:: console

      pip installe -e .[dev,doc]
