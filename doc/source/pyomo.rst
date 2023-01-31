.. _pyomo:

=========
Pyomo use
=========

This page aims to give some useful information about the use of pyomo module in
:term:`hots`.

What is pyomo ?
===============

Pyomo is a Python-based open-source software package that supports a diverse
set of optimization capabilities for formulating, solving, and analyzing
optimization models. See `the Pyomo home page <https://pyomo.readthedocs.io>`_
for more information about it.

How is it used ?
================

The first layer accessing the optimization problem is in model.py, acting like
an interface for the considered optimization problem. 
Indeed, the full optimization problem is defined in another file. We provide
some examples of this file, defining a specific use case, but it is supposed
to be user-defined.

The module provide a :term:`ModelInterface` which will call the provided file
for building the wanted model. This building can be done following any method,
as long as it provides an instance (a ConcreteModel or an instance of
AbstractModel created with :term:`create_instance` method).

Some explanation about example
==============================

In the example :term:`model_small_pyomo.py`, we build a simple placement problem
using the pyomo built-in objects and functions. The comments in the code can
guide the user to understand how we build the model.
Then we build a data dictionary from data in .csv files, and create an instance
of our model using :term:`create_instance` method.

Here is the problem built in the example :

.. math::
    \begin{alignat}{3}
         & \!\min      & \qquad & \sum_{n=1}^{N}{a_{n}}\\
         & \text{s.t.} &        & cons_{n,t} \leq cap_{n},                      & \qquad \forall n \in N, t \in T, \\
         &             &        & \sum_{n=1}^{N}{x_{c,n}}= 1,                   & \qquad \forall c \in C,        \\
         &             &        & x_{c,n} \leq a_{n},                           & \qquad \forall c \in C, n \in N \\
         &             &        & x_{c,n} \in \{0,1\}, a_{n} \in \{0,1\}
    \end{alignat}

.. todo:: Adapt with last code version