.. _usermanual:

===========
User manual
===========

Preparing data
==============

Input data are provided in 3 CSV files hosted in the same directory:

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

Preparing the parameters
========================

The parameters inputs are provided from a JSON file, which has the following format :

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

There is 2 ways to specify the parameter file to use :
   - with the options :code:`--params` (or :code:`-p`)
   - by including a file named :file:`params.json` in the data folder

Here are all the possible parameters with a small description :

- :code:`analysis` parameters dealing with analysis period

  - :code:`window_duration` window size for the loop process
  - :code:`sep_time` time dividing data between analysis and running period

- :code:`clustering` parameters dealing with first clustering problem

  - :code:`algo` algorithm to use for first clustering (between `kmeans`, `hierarchical` and `spectral`)
  - :code:`nb_clusters` number of clusters to use

- :code:`data` parameters dealing with data 

  - :code:`individuals_file` filename for containers consumption 
  - :code:`hosts_meta_file` filename for nodes information
  - :code:`individual_field` field name for containers ID in data
  - :code:`host_field` field name for nodes ID in data
  - :code:`tick_field` field name for timestamps ID in data
  - :code:`metrics` resources to take into account from data

- :code:`heuristic` parameters dealing with placement heuristic during analysis period

  - :code:`algo` heuristic algorithm used to have first placement solution (between `distant_pairwise`, `ffd` and `spread`)

- :code:`optimization` parameters dealing with optimization models solve

  - :code:`model` path to file describing the models to use (see :ref:`pyomo` for more information)
  - :code:`solver` the solver to use for solving problems

- :code:`loop` parameters dealing with loop process

  - :code:`mode` triggering loop mode (between `event`, `sequential` and `hybrid`)
  - :code:`tick` number of datapoints used to progress in time before triggering a new loop
  - :code:`constraints_dual` list of constraints used for dual variables comparison during solutions evaluation
  - :code:`tol_dual_clust` tolerance threshold for dual variables comparison during clustering evaluation
  - :code:`tol_move_clust` maximum allowed moves for clustering update
  - :code:`tol_dual_place` tolerance threshold for dual variables comparison during placeent evaluation
  - :code:`tol_move_place` maximum allowed moves for placement update
  - :code:`tol_step` tolerance increment factor for each loop

- :code:`plot` parameters dealing with graph display

  - :code:`renderer` rendering method used by matplotlib

- :code:`allocation` parameters dealing with adjustment of resources allocated to containers 

  - :code:`enable` enable or disable the dynamic adjustment of containers resources
  - :code:`constraints` constraints used for resources dynamic adjustment

    - :code:`load_threshold` maximum nodes load threshold  
    - :code:`max_amplitude` maximum nodes resource consumption amplitude 

  - :code:`objective` allocation problem objectives

    - :code:`open_nodes` number of used nodes
    - :code:`target_load_CPU` nodes load (CPU)

- :code:`placement` parameters dealing with containers placement problem

  - :code:`enable` enable or disable the containers placement problem

An parameter example file can be found in  :file:`~/tests/params_default.json` file.
Note that if no parameter file is provided, this example parameter file will be used.

Running the app
===============

Having the first 3 above mentioned files in an arbitrary directory - say :file:`~/path/to/data/` -
issue the command:

.. code:: console

   hots ~/path/to/data/

The :code:`hots` can be used with the following options :

- :code:`-k` : number of clusters used in clustering
- :code:`-t, --tau` : window size for the loop process
- :code:`-m, --method` : global method used for placement problem
- :code:`-c, --cluster_method` : method used to update the clustering
- :code:`-p, --param` : specific parameters file
- :code:`-o, --output` : specific directory for --output
- :code:`-ec, --tolclust` : value for epsilonC (building the conflict graph for clustering)
- :code:`-ea, --tolplace` : value for epsilonA (building the conflict graph for placement)
- :code:`--help` : display these options and exit

Note that some parameters can be redundant with the parameter file (e.g. :code:`k` and :code:`tau`)
: in this case the value from CLI is used. 

Reading the results
===================

When the application is launched, the whole initial data is displayed :

- the container resource usage
- the node resource usage (based on initial allocation)

The separation time (between the two phases) is plotted by a red line.

Then the first part of the methodology is performed (clustering on first time
period), and the allocation resulting from heuristic applied. The clustering
results and new nodes resource usage (based on new allocation) are displayed.

Finally, clustering results, containers and nodes consumptions are plotted and
updated in time, for the second phase.

.. todo:: Explain what happens and how to read the various figures that raise in new windows.
