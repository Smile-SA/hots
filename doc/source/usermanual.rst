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

.. todo:: Using the application in a Docker container.

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
