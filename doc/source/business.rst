.. _business:

============
Business use
============

As described in :ref:`introduction`, :term:`hots` is meant to handle time series based environments
and optimize these time series management. We tried to remain as generic and as adaptable as
possible, but :term:`hots` has been developped dealing with the specific containers resource
allocation context. This genericity effort can be seen through two main parts, which can be easily
adapted by users to tackle their specific use case.

Optimization models modularity
==============================

In order to evaluate the current solutions, :term:`hots` use some optimization models to find the
way to update these solutions (more details in :ref:`pyomo`). In the package, the currently used
optimization models are described in the :file:`model.py` and tackle two problems :

  - the clustering problem
  - the containers placement problem, using clustering information

The user has the possibility to provide his own file (or modifying the provided one), in order to
adapt the optimization constraints, variables and / or objectives to his use case, being related
to containers placement or not.

Business components modularity
==============================

Aside from the optimization part, the functions developped for handling the business use case (in
our case the container resource allocation) are grouped in a specific file. For example, in the
analysis period, several heuristics have been developped in order to propose a first containers
placement solution with first historical data : these heuristics are found in the
:file:`placement.py` file.

Besides the placement problem, we wanted to tackle the problem of resources allocated to containers
: a new file :file:`allocation.py` has been created and linked to the :file:`main.py` file. It
shows the possibility for the user to use either alternatives for heuristics developped in
:term:`hots`, or one full new use case, defining the algorithms and the optimization models use.