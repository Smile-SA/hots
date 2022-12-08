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

Note that the file `node_usage.csv` is not mandatory : if it does not exist in
the directory, it will be built with `container_usage.csv` data.

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


Running the app
===============

Having the first 3 above mentioned files in an arbitrary directory - say :file:`~/data/`,
and the parameter file like :file:`~/params.json` issue the command:

.. code:: console

   hots --data=~/work --params=~/params.json

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
