.. _outputs:

===================
Outputs explanation
===================

With the execution of HOTS, the following output and logs files are created:

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