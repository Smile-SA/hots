.. _install:

============
Installation
============

Requirements
============

:term:`hots` works with Python 3.11 to 3.13 on any \*nix-like OS. A graphic display is only
needed when the :code:`visualization` section of the configuration is enabled.

Due to some needed packages installation, the `dev` Python version is required. Make sure to install
the `dev` version of Python you will use. For example, if you want to use Python 3.12, you can
install the package `python3.12-dev`.

:term:`hots` uses a solver in order to solve some optimization problems, through the use of `Pyomo`
(see :ref:`pyomo` for more information about its use). The user can use any solver working with
`Pyomo`, but this solver needs to be installed and indicated to `Pyomo`. By default, :term:`hots`
uses `GLPK`, an open-source solver. `GLPK` is not a Python package, so it is not installed with
:term:`hots`: install the following packages (through `apt` for example):

* `libglpk-dev`
* `glpk-utils`

Production
==========

.. note::

   This operating mode is intended for end users.

.. code:: console

   pip install hots

To use the Kafka connector, install the :code:`kafka` extra instead:

.. code:: console

   pip install "hots[kafka]"

Check the installation with :code:`hots --version`.

.. hint::

   Install :term:`hots` in a virtual environment rather than system-wide, for example with
   ``python3 -m venv .venv && . .venv/bin/activate``.

Development
===========

.. note::

   This operating mode is intended for application maintainers.

.. note::

   You can find a `Makefile` that creates a virtual environment and install all the needed packages
   and hots. If you do that, do not forget to activate the virtual environment before running `hots`.

You should create and activate a dedicated Python virtual environment with whatever tool you prefer
(virtualenv, venv, pew, pyenv, ...). The instruction that follow assume this virtual environment is
activated.

Clone the source code from the |vcs_server|, then:

.. code:: console

   cd /where/you/cloned/hots
   pip install -e .[dev]

This last command installs the :command:`hots` command and the :term:`hots` package in the virtual
environment in "editable" mode. This means that every change you make in the Python source code is
immediately active (you don't need to reinstall after each change).

.. hint::

   If you want to use Kafka for data processing, you have to replace the last command with:

   .. code:: console

      pip install -e .[dev,kafka]

.. hint::

   If you need to change this documentation too, you may rather replace the last command with:

   .. code:: console

      pip install -e .[dev,doc]

Docker
======

You can also use Docker to install and run :term:`hots`.  
If you are not used to Docker, you can follow the installation guideline here : https://docs.docker.com/engine/install/, and the post-install process here (Linux) : https://docs.docker.com/engine/install/linux-postinstall/.

As soon as Docker is setup, you can run the following commands (being at the root of the directory, with the Dockerfile) :

.. code:: console

   docker build -t hots .

The image runs :term:`hots` by default, so arguments go straight to it (see section
:ref:`usermanual`):

.. code:: console

   docker run --rm hots --version

To get a shell instead, in which :code:`hots` is on the :code:`PATH`:

.. code:: console

   docker run -it --rm --entrypoint /bin/bash hots
