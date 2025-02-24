.. _usermanual:

===========
User manual
===========

Preparing the parameters
========================

The configuration file is the only mandatory argument to run HOTS. It is provided
through a .JSON file, which has the following format :

.. code-block:: JSON

   {
      "analysis": {
         "window_duration": "default",
         "step": 1
      },
      "clustering": {
         "algo": "hierarchical",
         "nb_clusters": 4
      },
      "heuristic": {
         "algo": "distant_pairwise"
      }
   }

Here are all the possible parameters with a small description :

- :code:`csv` use historical data as CSV or not

- :code:`host_meta_path` path to node capacity if needed

- :code:`data` parameters dealing with data 

  - :code:`individuals_file` filename for containers consumption 
  - :code:`hosts_meta_file` filename for nodes information
  - :code:`individual_field` field name for containers ID in data
  - :code:`host_field` field name for nodes ID in data
  - :code:`tick_field` field name for timestamps ID in data
  - :code:`metrics` resources to take into account from data

- :code:`analysis` parameters dealing with analysis period

  - :code:`sep_time` time dividing data between analysis and running period
  - :code:`placement_recompute` compute a placement solution in analysis period

- :code:`clustering` parameters dealing with first clustering problem

  - :code:`algo` algorithm to use for first clustering (between `kmeans`, `hierarchical` and `spectral`)
  - :code:`nb_clusters` number of clusters to use

- :code:`heuristic` parameters dealing with placement heuristic during analysis period

  - :code:`algo` heuristic algorithm used to have first placement solution (between `distant_pairwise`, `ffd` and `spread`)

- :code:`optimization` parameters dealing with optimization models solve

  - :code:`solver` the solver to use for solving problems
  - :code:`verbose` display all solving information

- :code:`loop` parameters dealing with loop process

  - :code:`mode` triggering loop mode (between `event`, `sequential` and `hybrid`)
  - :code:`window_duration` window size for the loop process
  - :code:`tick` number of datapoints used to progress in time before triggering a new loop
  - :code:`constraints_dual` list of constraints used for dual variables comparison during solutions evaluation
  - :code:`tol_dual_clust` tolerance threshold for dual variables comparison during clustering evaluation
  - :code:`tol_move_clust` maximum allowed moves for clustering update
  - :code:`tol_dual_place` tolerance threshold for dual variables comparison during placeent evaluation
  - :code:`tol_move_place` maximum allowed moves for placement update
  - :code:`tol_step` tolerance increment factor for each loop

- :code:`placement` parameters dealing with container placement problem

  - :code:`enable` enable or disable the placement mechanism
  - :code:`allocation_support` enable or disable support for resource allocation
  - :code:`allocation_objectives` objectives related to resource allocation

    - :code:`open_nodes` number of used nodes

- :code:`allocation` parameters dealing with adjustment of resources allocated to containers 

  - :code:`enable` enable or disable the dynamic adjustment of containers resources
  - :code:`constraints` constraints used for resources dynamic adjustment

    - :code:`load_threshold` maximum nodes load threshold  
    - :code:`max_amplitude` maximum nodes resource consumption amplitude 

  - :code:`objective` allocation problem objectives

    - :code:`open_nodes` number of used nodes
    - :code:`target_load_CPU` nodes load (CPU)

- :code:`kafkaConf` Kafka configuration parameters

  - :code:`topics` Kafka topics used for message exchange

    - :code:`docker_topic` topic name for Docker placement
    - :code:`docker_replacer` topic name for Docker replacement testing
    - :code:`mock_topic` topic name for mock placement

  - :code:`Producer` Kafka producer configuration

    - :code:`brokers` list of broker addresses for producing messages

  - :code:`Consumer` Kafka consumer configuration

    - :code:`group` consumer group name
    - :code:`brokers` list of broker addresses for consuming messages

  - :code:`schema` Avro schema definition for container data

    - :code:`type` schema type
    - :code:`name` schema name
    - :code:`namespace` schema namespace
    - :code:`fields` list of fields in the schema

      - :code:`containers` array of container records

        - :code:`timestamp` integer representing the timestamp
        - :code:`container_id` string representing the container identifier
        - :code:`machine_id` string representing the machine identifier
        - :code:`cpu` float representing CPU usage

  - :code:`schema_url` URL of the schema registry

A parameter example file can be found in  :file:`~/tests/data/thesis_ex_10/params.json` file.

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

HOTS can be run using the following command:

.. code:: console

   hots ~/path/to/config/file

The :code:`hots` can be used with the following options :

- :code:`-k` : number of clusters used in clustering
- :code:`-t, --tau` : window size for the loop process
- :code:`-m, --method` : global method used for placement problem
- :code:`-c, --cluster_method` : method used to update the clustering
- :code:`-o, --output` : specific directory for --output
- :code:`-C, --tolclust` : value for epsilonC (building the conflict graph for clustering)
- :code:`-A, --tolplace` : value for epsilonA (building the conflict graph for placement)
- :code:`-K, --use_kafka` : use Kafka streaming platform for data processing
- :code:`-T, --time_limit` : Provide a time limit for data processing (in seconds)
- :code:`--help` : display these options and exit

Note that some parameters can be redundant with the parameter file (e.g. :code:`k` and :code:`tau`)
: in this case the value from CLI is used. 

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
