.. _usermanual:

===========
User manual
===========

Preparing the parameters
========================

The configuration file is the only mandatory argument to run :term:`hots`. It is a JSON
file passed to the CLI with the :code:`--config` option.

A minimal example using the file connector is the following:

.. code-block:: json

   {
     "time_limit": null,
     "clustering": {
       "method": "kmeans",
       "nb_clusters": 3,
       "parameters": {
         "nb_clusters": 3
       }
     },
     "optimization": {
       "backend": "pyomo",
       "parameters": {
         "solver": "glpk",
         "verbose": 0
       }
     },
     "problem": {
       "type": "placement",
       "parameters": {
         "initial_placement": 0,
         "tol": 0.1,
         "tol_move": 0.5
       }
     },
     "connector": {
       "type": "file",
       "parameters": {
         "data_folder": "./tests/data/thesis_ex_10",
         "file_name": "container_usage.csv",
         "host_field": "machine_id",
         "individual_field": "container_id",
         "tick_field": "timestamp",
         "tick_increment": 2,
         "window_duration": 3,
         "sep_time": 3,
         "metrics": ["cpu"],
         "outfile": "./out/moves.log"
       }
     },
     "logging": {
       "level": "INFO",
       "filename": "./out/hots.log",
       "fmt": "%(asctime)s %(levelname)s: %(message)s"
     },
     "reporting": {
       "results_folder": "./out",
       "metrics_file": "./out/metrics.csv",
       "plots_folder": "./out/plots"
     }
   }

Configuration sections
----------------------

The top-level keys of the configuration file map to the fields of
:class:`hots.config.loader.AppConfig`:

- :code:`time_limit`:
  maximum wall-clock time (in seconds) for the application run.
  If :code:`null`, :term:`hots` processes all available data.

- :code:`clustering`:
  configuration of the clustering plugin.

  - :code:`method`: name of the clustering algorithm to use, e.g.
    :code:`"kmeans"`, :code:`"hierarchical"`, :code:`"spectral"`.
  - :code:`nb_clusters`: target number of clusters.
  - :code:`parameters`: free-form dict of method-specific parameters.

- :code:`optimization`:
  configuration of the optimization backend.

  - :code:`backend`: optimization backend to use. Currently :code:`"pyomo"` is
    supported and maps to :class:`hots.plugins.optimization.pyomo_model.PyomoModel`.
  - :code:`parameters`: solver-related parameters such as:

    - :code:`solver`: solver name (e.g. :code:`"glpk"`).
    - :code:`verbose`: integer verbosity level.

- :code:`problem`:
  configuration of the business (domain) problem plugin (see
  :mod:`hots.plugins.problem.placement`).

  - :code:`type`: problem type. The default implementation is :code:`"placement"`.
  - :code:`parameters`: problem-specific parameters, for example:

    - :code:`initial_placement`: whether to compute an initial placement
      (:code:`0` or :code:`1`).
    - :code:`tol`: tolerance used to decide when a node is overloaded.
    - :code:`tol_move`: tolerance for deciding when to move a container.

- :code:`connector`:
  configuration of the data connector plugin.

  - :code:`type`: connector type. The built-in types are:

    - :code:`"file"`: read data from CSV files.
    - :code:`"kafka"`: read data from a Kafka topic.

  - :code:`parameters`: connector-specific parameters. For both connectors,
    the following keys are usually required:

    - :code:`data_folder`: folder containing the input data.
    - :code:`file_name`: CSV file containing container-level metrics.
    - :code:`individual_field`: column name for container IDs.
    - :code:`host_field`: column name for host (node) IDs.
    - :code:`tick_field`: column name for timestamps.
    - :code:`tick_increment`: step between two timestamps in the stream.
    - :code:`window_duration`: window size used for sliding-window analysis.
    - :code:`sep_time`: time separating the analysis period from the running period.
    - :code:`metrics`: list of metric names to use (e.g. :code:`["cpu"]`).
    - :code:`outfile`: file where proposed moves will be written.

    For the Kafka connector, the following additional keys can be used:

    - :code:`bootstrap.servers`: Kafka bootstrap servers string.
    - :code:`topics`: list of topic names used to publish moves.
    - :code:`connector_url`: optional URL of an external connector service.

- :code:`logging`:
  logging configuration used by :func:`hots.utils.logging_config.setup_logging`.

  - :code:`level`: log level (:code:`"DEBUG"`, :code:`"INFO"`, ...).
  - :code:`filename`: log file path.
  - :code:`fmt`: log message format.

- :code:`reporting`:
  configuration of result files and plots.

  - :code:`results_folder`: base output folder.
  - :code:`metrics_file`: CSV file for aggregated metrics.
  - :code:`plots_folder`: folder where plots will be saved.

Additional top-level keys
-------------------------

For convenience, some examples also define :code:`data_folder`,
:code:`individual_field`, :code:`host_field`, :code:`tick_field` and
:code:`metrics` at the top level. These are redundant copies of the values
stored inside :code:`connector.parameters` and are kept for backward
compatibility.

Preparing data
==============

If you use historical data, the inputs are provided through 3 CSV files hosted in the same directory:

- :file:`container_usage.csv` : describes containers resource consumption
- :file:`node_meta.csv` : provides nodes capacities (and other additional data)
- :file:`node_usage.csv` : describes nodes resource consumption

Each file have the following formats :

- :file:`container_usage.csv` :
   .. csv-table::
      :header: "timestamp", "container_id", "metric_1", "metric_2", "machine_id"
      :widths: 15, 15, 15, 15, 15

      "t1", "c_10", 10, 50, "m_2"
      "...", "...", "...", "...", "..."
      "tmax", "c_48", 6.5, 24, "m_5"
- :file:`node_meta.csv` :
   .. csv-table::
      :header: "machine_id", "metric_1", "metric_2"
      :widths: 15, 15, 15

      "m_2", 30, 150
      "m_5", 24, 80
- :file:`container_usage.csv` :
   .. csv-table::
      :header: "timestamp", "machine_id", "metric_1", "metric_2"
      :widths: 15, 15, 15, 15

      "t1", "m_2", 25, 65
      "...", "...", "...", "..."
      "tmax", "m_5", 17.5, 52

Note that the file :file:`node_usage.csv` is not mandatory : if it does not exist in
the directory, it will be built using :file:`container_usage.csv` data.

Running the app
===============

Once the configuration file is prepared, :term:`hots` is started by:

.. code:: console

   hots --config /path/to/config.json

The :code:`--config` (or :code:`-c`) option is mandatory and must point to a valid
JSON configuration file.

You can view the available command-line options with:

.. code:: console

   hots --help

Typical options include the standard :code:`--help` and :code:`--version` flags.
All runtime behaviour is controlled by the configuration file rather than
individual CLI flags (number of clusters, window size, problem type, connector,
etc.).

Output explanation
===================

With the execution of HOTS, the global process is displayed in the terminal and
the following output and logs files are created:

* :file:`logs.log`: logs on main process (which loop, which step in the loop...)
* :file:`clustering_logs.log`: logs on clustering computes at each loop
* :file:`optim_logs.log`: information on optimization models solving
* :file:`results.log`: temporary results at each loop (number of changes, objective value...)
* :file:`global_results.csv`: final results for identified business criteria 
* :file:`loop_results.csv`: multiple indicators at each loop (clustering criteria, conflict graph information...)
* :file:`node_results.csv`: final nodes related results (average / minimum / maximum loads)
* :file:`times.csv`: intermediate times for each step (preprocess + all steps for each loop)
* :file:`node_usage_evo.csv`: numerical nodes consumption evolution, since HOTS launch until HOTS stop
* :file:`node_usage_evo.svg`: graphical nodes consumption evolution, since HOTS launch until HOTS stop
