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

In :term:`hots` package, all the Pyomo related work is located in :file:`model.py`. Pyomo allows us
to create optimization models (see section :ref:`process` for more details about the optimization
models use), in our case modeling the clustering and the containers placement problem.

We define an abstraction of the optimization model and their use at a higher level, allowing the
user to customize the optimization part as he wants, either by providing small changes on
optimization models in the same file or bigger changes, like completely different models and / or
different use, defined in his own file.

The optimization models need a solver to be solved, and the use of Pyomo allows a wide variety of
solvers to be used (`see here
<https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html#supported-solvers>`_). In
:term:`hots`, this solver is simply indicated in the parameter file (see section
:ref:`usermanual`).

Inside the :file:`pyomo.py` file, all the model definition and creation (described next) are
specified inside a class named :file:`Model`. All the functions outside this class define how these
models are used. 

Optimization models description
===============================

In the :file:`model.py` provided file, 2 optimization models are created, representing our use case
: the clustering problem and the placement problem.

The process followed in the :file:`model.py` file to build the optimization models is as follow :

#. Create an Pyomo abstract model to handle your model at a high level 
#. Define the parameters that will help building the abstract model (objects from which variables and constraints will be created)
#. Define the variables, constraints and objective defining your model (the provided models are shown as examples at the end of this section)
#. Create data objects that will define the model instance (object from which the data will be taken)
#. Create a Pyomo instance, instantiating variables, constraints and objective using data objects.

See the Pyomo documentation for more details about all these objects.

Finally, the clustering optimization model provided in the :file:`model.py` is as follow :

.. math::
    \begin{alignat}{3}
        & \!\min      & \qquad & \sum_{(i,j) \in [1..I]}{w_{i,j}u_{i,j}}   &                                    \\
        & \text{s.t.} &        & \sum_{k=1}^{K}{y_{i,k}}= 1                & \forall i \in [1..I]               \\
        &             &        & y_{i,k} \leq b_{k}                        & \forall i \in [1..I], k \in [1..K] \\
        &             &        & \sum_{k=1}^{K}{b_{k}} \leq nb\_clusters   &                                    \\
        &             &        & {u_{i,j}} =\sum_{k=1}^{K}{y_{i,k}y_{j,k}} & \forall (i,j) \in [1..I]
    \end{alignat}

The used placement optimization model is the following :

.. math::
    \begin{alignat}{3}
        & \!\min      & \qquad & \sum_{(i,j) \in [1..I]}{{u_{i,j}}{v_{i,j}} + (1-{u_{i,j}})v_{i,j}d_{i,j}} &                                      \\
        & \text{s.t.} &        & {cons_{m,t}} \leq cap_{m}                                                 & \forall m \in [1..M], \ t \in [1..T] \\
        &             &        & \sum_{m=1}^{M}{x_{i,m}}= 1                                                & \forall i \in [1..I]                 \\
        &             &        & {x_{i,m}} \leq a_{m}                                                      & \forall i \in [1..I], \ m \in [1..M] \\
        &             &        & \sum_{m=1}^{M}{a_{m}} \leq max\_nodes                                     &                                      \\
        &             &        & {v_{i,j}}=\sum_{m=1}^{M}{x_{i,m}x_{j,m}} 
    \end{alignat}