.. _introduction:

====================
Introduction to hots
====================

:term:`hots` is an application for testing a hybrid resource allocation method using machine learning
and optimization.

Intended audience
=================

:term:`hots` is firstly thought for system and network administrators, allowing
them to test new resource allocation techniques for container based hosting systems.

It could be used in the future for other usage (still in resource allocation
context, or more extended), but at the moment the methodology is not generic
enough.

What is hots
============

The main goal of :term:`hots` is to optimize the resource usage in the context
we are. 

The first step is to get historical data of resource usage by individuals (like
CPU and/or memory usage by containers). So we have timeseries describing the
evolution of these consumptions. We perform a clustering on these timeseries
for grouping individuals into similar profile groups.

Then we apply a heuristic, developped using clustering information (like
consumption profiles) to *smartly* allocate individuals (containers) on the
nodes, trying to save resources and prevent from overflows or other problems.

Then we initiate a loop in which we retrieve new data as things progress (the
evolution of resource consumption), we check the validity of existing
clustering (update it if needed), and adapt the existing allocation depending
on the clustering evolution.

Usage example
=============

:term:`hots` has been developped using historical data (`.CSV` files) and can be
run alone in this case.
A streaming platform (`Kafka`) has been added to have streaming data incoming
(from historical data or from real environment). But in order to be used with a
rela environment, it has to be linked to a connector, that can send the data to
Kafka, and retrieve the moving container messages (and apply these moves).
The whole dataset (historical or real-time) has two period, based on time:

- the first part is used to have the first timeseries, on which we perform the clustering
- the second part is used for the *streaming* process, in which we retrieve data,
incremente timestamp, and perform the evaluation described above.

A precise run description is given in section :ref:`process`.

Moreover, as this project started with a PhD, you can find a lot of details in the thesis
manuscript `here <https://theses.hal.science/tel-03997934>`_ (state of the art, algorithms,
optimization models, benchmark ...).