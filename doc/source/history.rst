.. _history:

===============
Release history
===============

.. This section will enumerate all releases of hots in reverse chronological order. For each release,
.. provide:
..
.. - The version, and release date.
.. - The major changes provided by the release.

.. rubric:: 2.1.0 - Kafka connector for live environments

- New Kafka connector, used with `hots-connector <https://github.com/Smile-SA/hots-connector>`_.
  **Incompatible configuration change:** :code:`topics` is replaced by :code:`topic_data` and
  :code:`topic_moves`; new keys :code:`consumer_conf`, :code:`poll_timeout`,
  :code:`bootstrap_timeout_seconds` and :code:`node_capacities`. :code:`connector_url` is removed.
  The Kafka connector of 2.0.x returned no initial data, so no working 2.0.x Kafka setup is
  expected to break.
- New :code:`data` configuration section for the column names. The keys are still accepted in
  :code:`connector.parameters`. Only one metric is supported, and a longer list is rejected.
- Message contract with hots-connector, schema :code:`v: 1` (see :ref:`contract`).
- Placement, evaluation and Pyomo model fixes; the persistent Pyomo model is reused between ticks.
- :code:`hots --version`; the sample configurations ship with the package.
- Packaging: Python 3.11 to 3.13 (Python 3.10, end of life, is no longer supported); the wheel no
  longer installs :file:`VERSION.txt` at the root of the environment; the :code:`kafka` extra only
  installs :code:`confluent-kafka`.
- Docker image: Python 3.12 on Debian bookworm (was Python 3.10 on bullseye, both end of life);
  the image runs :code:`hots` directly from its virtual environment, which is on the
  :code:`PATH`.

.. rubric:: 2.0.3 - Plugin and configuration fixes, uv-based build and Docker image
.. rubric:: 2.0.2 - Improved tests
.. rubric:: 2.0.1 - Documentation build fix
.. rubric:: 2.0.0 - Reorganized configuration file, incoming containers handled during the run

.. rubric:: 1.0.1 - Refined plugin interfaces and minor fixes
.. rubric:: 1.0.0 - Major refactor: plugin-based architecture, new connector layer, revamped configuration format and CLI

.. rubric:: 0.1.5 - 09.03.2023 - Thesis reference and right docstrings added
.. rubric:: 0.1.4 - 01.03.2023 - Theme documentation changed
.. rubric:: 0.1.3 - 28.02.2023 - Documentation fulfilled
.. rubric:: 0.1.2 - 01.02.2023 - Package published on PyPi, GitHub, Documentation on ReadTheDocs 
.. rubric:: 0.1.1 - 20.01.2023 - Publishing preparation
.. rubric:: 0.1.0 - 10.01.2023 - Pyomo working

Initial release
