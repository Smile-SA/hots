.. _introduction:

===================
Introduction to rac
===================

:term:`rac` is an application for testing a hybrid resource allocation method using machine learning
and optimization.

Intended audience
=================

:term:`rac` is firstly thought for system and network administrators, allowing
them to test new resource allocation techniques for container based hosting systems.

It could be used in the future for other usage (still in resource allocation
context, or more extended), but at the moment the methodology is not generic
enough.

What is rac
===========

The main goal of :term:`rac` is to optimize the resource usage in the context
we are. 

The first step is to get historical data of resource usage by individuals (like
CPU and/or memory usage by containers). So we have timeseries describing the
evolution of these consumptions. We perform a clustering on these timeseries
for grouping individuals into similar profile groups.

Then we apply a heuristic, developped using clustering information (like
consumption profiles) for *smartly* allocate individuals (containers) on the
nodes, trying to save resources and prevent from overflows or other problems.

Then we initiate a loop in which we retrieve new data as things progress (the
evolution of resource consumption), we check the validity of existing
clustering (update it if needed), and adapt the existing allocation depending
on the clustering evolution.

Usage example
=============

For the moment, :term:`rac` is an evaluation of a methodology, then it does
not work with a *real* environment, but with a dataset of historical resource
usage data. In order to simulate the method, we split the dataset in two,
based on the time :

- the first half is used to have the first timeseries, on which we perform
    the clustering
- the second half is used to *simulate* the *streaming* process, in which
    we retrieve data, incrementing timestamp, and perform the evaluation
    described above.