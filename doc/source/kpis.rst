.. _kpis:

===================
HOTS KPIs Reference
===================

This page describes the performance indicators (KPIs) emitted by the HOTS
evaluation pipeline.

KPIs are produced **per step** (typically per streaming window / tick) as a flat
dictionary of scalar values.

Conventions
===========

- A *step* corresponds to one iteration of the scheduling/evaluation loop
  (e.g. one tick or window).
- KPI keys are grouped using prefixes:

  - ``clst__*``        : clustering change (vs previous step)
  - ``clst_struct__*`` : clustering structure at current step
  - ``plc__*``         : placement change (vs previous step)
  - ``plc_delta__*``   : placement change counts (vs previous step)
  - ``moves__*``       : moves returned / applied by ``adjust()``
  - ``clust_conf__*``  : conflict graph KPIs for clustering stage
  - ``place_conf__*``  : conflict graph KPIs for placement stage
  - ``load__*``        : host-load / balance KPIs from the working window

Some KPIs may be missing for some steps (for example, the first step has no
previous state; some conflict graphs may have no weights).

------------------------------------------------------------

Step Metadata
=============

These fields are not KPIs strictly speaking, but they are strongly recommended
to be present in every metrics row to enable proper analysis.

- ``run_id`` (str)
  Unique identifier for the run (e.g. timestamp-based).

- ``loop_nb`` (int)
  Index of the evaluation loop step.

- ``tick`` (int)
  Tick identifier for the step (if using a single-tick model).

- ``tick_start`` (int)
  Start tick of the window (if using windows).

- ``tick_end`` (int)
  End tick of the window (if using windows).

- ``window_duration`` (int or float)
  Window length in ticks or time units.

------------------------------------------------------------

``clst__*`` — Clustering Change KPIs
====================================

These KPIs compare cluster assignments between consecutive steps.

Let:

- ``labels_now[i]`` be the cluster label of individual ``i`` at the current step
- ``labels_prev[i]`` be the cluster label of individual ``i`` at the previous step

Metrics - Clustering Change
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``clst__changed_count`` (float)

  Number of individuals whose cluster assignment changed compared to the
  previous step.

  Definition::

      sum_i 1(labels_now[i] != labels_prev[i])

- ``clst__changed_ratio`` (float)

  Fraction of individuals whose cluster assignment changed.

  Definition::

      changed_count / n

Interpretation - Clustering Change
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

High values indicate unstable or frequently changing clustering across time.

------------------------------------------------------------

``clst_struct__*`` — Clustering Structure KPIs
==============================================

These KPIs describe the cluster-size distribution at the current step.

Let the cluster sizes be ``{s_1, s_2, ..., s_k}``, where ``k`` is the number of
clusters.

Metrics - Clustering Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``clst_struct__n_clusters`` (float)

  Number of distinct clusters (``k``).

- ``clst_struct__singleton_ratio`` (float)

  Fraction of clusters of size 1.

  Definition::

      (# clusters with size 1) / k

- ``clst_struct__size_min`` (float)
- ``clst_struct__size_max`` (float)
- ``clst_struct__size_mean`` (float)
- ``clst_struct__size_std`` (float)

  Descriptive statistics of cluster sizes.

- ``clst_struct__size_entropy`` (float)

  Shannon entropy (natural logarithm) of the cluster-size distribution.

  Higher values indicate more balanced cluster sizes.

- ``clst_struct__size_gini`` (float)

  Gini coefficient of cluster sizes.

  ``0`` means perfectly equal sizes; higher values indicate inequality.

Interpretation - Clustering Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- High ``singleton_ratio`` suggests over-fragmentation.
- High ``size_gini`` indicates a few large clusters and many small ones.

------------------------------------------------------------

``plc__*`` — Placement Change KPIs
==================================

These KPIs compare placement (host assignment) between consecutive steps.

Let:

- ``placement_now[id]`` be the assigned host for individual ``id`` at the current step
- ``placement_prev[id]`` be the assigned host at the previous step

Metrics - Placement Change
^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``plc__moved_count`` (float)

  Number of individuals whose host assignment changed.

- ``plc__moved_ratio`` (float)

  Fraction of individuals whose host assignment changed.

  Definition::

      moved_count / n_common

Interpretation - Placement Change
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

High values indicate placement churn or instability.

------------------------------------------------------------

``plc_delta__*`` — Placement Delta Counts
=========================================

Simple counts derived from comparing placements between steps.

Metrics - Placement Delta
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``plc_delta__moved_count`` (float)

  Number of individuals whose host assignment changed.

- ``plc_delta__stable_count`` (float)

  Number of individuals whose host assignment remained unchanged.

Interpretation - Placement Delta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Useful for sanity checks and debugging placement behavior.

------------------------------------------------------------

``moves__*`` — Applied Moves KPIs
=================================

These KPIs summarize the result of ``problem.adjust(...)``.

Metrics - Applied Moves
^^^^^^^^^^^^^^^^^^^^^^^

- ``moves__moves_count`` (float)

  Number of moves applied or returned by ``adjust()``.

- ``moves__moves_ratio`` (float)

  Moves normalized by population size.

  Definition::

      moves_count / n_indiv

Interpretation - Applied Moves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Comparing these KPIs with ``plc__moved_*`` can reveal differences between
explicit adjustments and final placement changes.

------------------------------------------------------------

Conflict Graph KPIs
===================

Conflict graphs represent coupling between individuals derived from step-to-step
signals (e.g. dual changes).

Two sets of KPIs are produced:

- ``clust_conf__*`` : conflict graph from clustering stage
- ``place_conf__*`` : conflict graph from placement stage

Let:

- ``n`` be the number of nodes
- ``m`` be the number of edges

Graph Structure Metrics
^^^^^^^^^^^^^^^^^^^^^^^

- ``*_conf__nodes`` (float)
- ``*_conf__edges`` (float)

- ``*_conf__density`` (float)

  Graph density::

      m / (n * (n - 1) / 2)   (for n > 1)

Connectivity Metrics
^^^^^^^^^^^^^^^^^^^^

- ``*_conf__components`` (float)

  Number of connected components.

- ``*_conf__largest_component_ratio`` (float)

  Size of the largest connected component divided by ``n``.

Degree Statistics
^^^^^^^^^^^^^^^^^

- ``*_conf__deg_mean`` (float)
- ``*_conf__deg_std`` (float)
- ``*_conf__deg_p95`` (float)
- ``*_conf__deg_max`` (float)

Edge Weight Statistics (if weights exist)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``*_conf__w_sum`` (float)
- ``*_conf__w_mean`` (float)
- ``*_conf__w_p95`` (float)
- ``*_conf__w_max`` (float)

Interpretation - Conflict graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

High density, large components, or high-degree nodes indicate strong coupling
between decisions and may correlate with instability or reconfiguration cost.

------------------------------------------------------------

``load__*`` — Host Load / Balance KPIs
======================================

These KPIs summarize how a chosen metric (e.g. CPU usage) is distributed across
hosts during the current working window.

Let:

- ``L_h`` be the aggregated load of host ``h`` over the window

Metrics - Host
^^^^^^^^^^^^^^

- ``load__n_hosts`` (float)

  Number of hosts observed.

- ``load__n_indiv`` (float)

  Number of individuals observed.

- ``load__host_load_mean`` (float)
- ``load__host_load_std`` (float)

  Mean and standard deviation of ``{L_h}``.

- ``load__host_load_cv`` (float)

  Coefficient of variation::

      std / mean   (0 if mean == 0)

- ``load__host_load_p95`` (float)
- ``load__host_load_max`` (float)

- ``load__host_load_gini`` (float)

  Gini coefficient of host loads.

Interpretation - Host
^^^^^^^^^^^^^^^^^^^^^

- High ``cv`` or ``gini`` indicates poor load balance.
- Tracking these KPIs over time helps assess scheduling effectiveness.

------------------------------------------------------------

Practical Notes
===============

- The first step of a run may not contain change-based KPIs
  (``clst__*``, ``plc__*``).
- Edge-weight KPIs may be zero if conflict graphs are unweighted.
- Prefer storing only scalar KPIs in the main metrics table.
  Detailed event data (e.g. lists of moved containers) should be stored
  separately (e.g. JSONL).
