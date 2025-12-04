.. _streaming:

==============
Streaming data
==============

This document describes how streaming data is integrated into :term:`hots` in
the refactored architecture.

Overview
========

Streaming is handled by *connector plugins* implementing the
:class:`hots.core.interfaces.ConnectorPlugin` interface. A connector is
responsible for:

* loading the initial data used during the analysis phase,
* providing new data for each step of the streaming loop,
* applying the moves decided by the optimization/heuristics layer (for instance
  by sending them to Kafka or writing them to a file).

The connector to use is selected via the :code:`connector` section of the
configuration file (see :ref:`usermanual`).

Connector plugin interface
==========================

The :class:`hots.core.interfaces.ConnectorPlugin` class defines the methods that
every connector must implement:

* :meth:`load_initial`: load the initial batch of data and return the
  individual-level dataframe, the host-level dataframe, and optional metadata.

* :meth:`next_batch`: return the next batch of individual-level data for the
  current streaming window or :code:`None` when no more data is available.

* :meth:`apply_moves`: apply a list of move dictionaries produced by the
  placement plugin. For example, a move can be represented as::

      {"container_name": "c_001", "old_host": "h_01", "new_host": "h_03"}

  The connector decides how to persist or forward these moves (to a file, a
  Kafka topic, an external orchestrator, etc.).

Built-in connectors
===================

File connector
--------------

:mod:`hots.plugins.connector.file_connector` implements the
:class:`hots.plugins.connector.file_connector.FileConnector` class, which:

* reads historical data from CSV files located in :code:`data_folder`,
* exposes them as :class:`pandas.DataFrame` objects to the rest of the
  application,
* simulates a streaming process by returning successive time windows in
  :meth:`next_batch`,
* writes computed moves to a log file, using the :code:`outfile` path provided
  in :code:`connector.parameters`.

Kafka connector
---------------

:mod:`hots.plugins.connector.kafka_connector` implements the
:class:`hots.plugins.connector.kafka_connector.KafkaConnector` class, which:

* consumes container usage data from one or several Kafka :code:`topics`,
* keeps track of offsets for robust consumption,
* converts messages into :class:`pandas.DataFrame` rows compatible with the rest
  of the pipeline,
* publishes move messages back to Kafka in :meth:`apply_moves`.

The Kafka connection details (bootstrap servers, topic names, etc.) are
configured through the :code:`connector.parameters` and optional top-level
:code:`kafka` section of the configuration file.

Integration in the main loop
============================

The main application class :class:`hots.core.app.App` interacts only with the
connector interface. This means you can implement your own connector (for
instance, to integrate with another streaming platform or REST API) by
subclassing :class:`ConnectorPlugin` and configuring :code:`connector.type`
accordingly, without changing the rest of the code base.
