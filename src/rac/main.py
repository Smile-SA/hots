"""
=========
rac main
=========
Entry point of rac module through ``rac data``.

    - data is the folder where we find the files
        x container_usage.csv : describes container resource consumption
        x node_meta.csv : describes nodes capacities
        (x node_usage.csv : describes nodes resource consumption)

The entire methodology is called from here (initialization, clustering,
allocation, evaluation, access to optimization model...).
"""

import time

import click

from matplotlib import pyplot as plt

# Personnal imports
from . import allocation as alloc
from . import clustering as clt
from . import container as ctnr
from . import instance
from . import model
from . import plot


# TODO do an entire loop, only some plot which last in loop

# Global variables
# {kmeans, hierarchical, spectral, spectral_perso}
clustering_algo = 'hierarchical'


@click.command()
@click.argument('data', type=click.Path(exists=True))
def main(data):
    """Perform all things of methodology."""
    #####################################################################
    # Initialization part
    main_time = time.time()

    # Init containers & nodes data, then Instance
    myInstance = instance.Instance(data)

    building_cplex_time = time.time()
    cplex_model = model.CPX_Instance(myInstance)
    print('CPLEX model imported.')
    print('Building CPLEX model time : %fs' %
          (time.time() - building_cplex_time))
    # TODO fix this method
    # print('Objective of initial placement : ',
    #       cplex_model.get_obj_value_heuristic(myInstance))

    # Plot data (containers & nodes consumption)
    ctnr.plot_allData_allContainers(
        myInstance.df_containers, sep_time=myInstance.sep_time)
    plot.plot_containers_groupby_nodes(
        myInstance.df_containers,
        myInstance.df_nodes_meta.cpu.max(),
        myInstance.sep_time)

    plt.show(block=False)
    input('Press any key to continue ...')

    #####################################################################
    # Clustering part
    df_containers_clust = clt.build_matrix_indiv_attr(
        myInstance.df_containers.loc[
            myInstance.df_containers['timestamp'] <= myInstance.sep_time])

    clustering_time = time.time()

    labels_ = clt.perform_clustering(
        df_containers_clust, clustering_algo, myInstance.nb_clusters)
    df_containers_clust['cluster'] = labels_
    myInstance.nb_clusters = labels_.max() + 1

    # print('Clustering balance : ',
    #       df_containers_clust.groupby('cluster').count())

    # TODO improve this part (distance...)
    cluster_vars = clt.get_cluster_variance(
        myInstance.nb_clusters, df_containers_clust)
    cluster_profiles = clt.get_cluster_mean_profile(
        myInstance.nb_clusters,
        df_containers_clust,
        myInstance.window_duration)
    cluster_var_matrix = clt.get_sumCluster_variance(
        cluster_profiles, cluster_vars)
    # cluster_distance = clt.get_distanceCluster(
    #     myInstance, cluster_profiles)

    # KMedoids
    # TODO quid kmedoids ?
    # D = clt.p_dist(df_containers_clust, metric='euclidean')
    # M, C = clt.kMedoids(D, 4)

    print('Clustering computing time : %fs' % (time.time() - clustering_time))

    #####################################################################
    # Plot clustering
    do_plot_clustering = True
    if do_plot_clustering:
        # plot.plot_containers_clustering_together(df_containers_clust)
        plot.plot_clustering(df_containers_clust)
        plot.plot_cluster_profiles(cluster_profiles)

        # plot.plot_clustering_containers_byNode(
        #     myInstance, labels_)

    #####################################################################
    # Allocation
    allocation_time = time.time()
    containers_grouped = alloc.allocation_distant_pairwise(
        myInstance, cluster_var_matrix, labels_)

    # plot.plot_containers_groupby_nodes(myInstance.df_containers)

    # alloc.allocation_ffd(myInstance, cluster_vars, cluster_var_matrix,
    #                      labels_)

    print('Allocation time : %fs' % (time.time() - allocation_time))

    # TODO make comparison between heuristics

    # Plot node's data
    # nd.plot_allData_allNodes(df_nodes)
    plot.plot_containers_groupby_nodes(
        myInstance.df_containers,
        myInstance.df_nodes_meta.cpu.max(),
        myInstance.sep_time)
    # nd.plot_allData_allNodes_end(myInstance.df_nodes, myInstance.time)

    # Check if sum(nodes.conso) for all tick is same as previous
    # (global_max_cpu, global_max_mem) = nd.plot_total_usage(
    #     myInstance.df_nodes, 'Total conso on all nodes after new alloc')
    # print(global_max_cpu, global_max_mem)

    # nd.print_vmr(myInstance.df_nodes, myInstance.time, 2)
    # myInstance.instance_inFile_after(instance_file)
    # myInstance.nodes_state_inFile(instance_file)
    # nd.get_mean_consumption(myInstance.df_nodes)
    # nd.get_variance_consumption(myInstance.df_nodes)

    # plt.show(block=False)
    plt.show(block=True)

    #####################################################################
    # CPLEX evaluation part
    # Update the solution in CPLEX model
    cplex_model.set_X_from_df(myInstance)
    # cplex_model.get_obj_value_heuristic(myInstance)
    # cplex_model.print_sol_infos_heur()
    print('Adding constraints from heuristic ...')
    cplex_model.add_constraint_heuristic(containers_grouped, myInstance)

    print('Solving linear relaxation ...')
    # cplex_model.solve_relax()
    # model.print_all_dual(cplex_model.relax_mdl)
    # cplex_model.get_max_dual()

    #####################################################################
    # Evaluation period
    fig_cont, ax_cont = plot.init_containers_plot(
        myInstance.df_containers, myInstance.sep_time)
    fig_node, ax_node = plot.init_nodes_plot(
        myInstance.df_containers, myInstance.sep_time)
    fig_clust, ax_clust = plot.init_plot_clustering(df_containers_clust)

    tmin = myInstance.df_containers['timestamp'].min()
    tmax = myInstance.sep_time

    end = False

    while not end:
        working_df = myInstance.df_containers[
            (myInstance.df_containers['timestamp'] >= tmin) &
            (myInstance.df_containers['timestamp'] <= tmax)]
        df_clust = clt.build_matrix_indiv_attr(working_df)
        df_clust['cluster'] = labels_
        cluster_profiles = clt.get_cluster_mean_profile(
            myInstance.nb_clusters, df_clust,
            myInstance.window_duration, tmin)
        plot.update_clustering_plot(fig_clust, ax_clust, df_clust)
        clt.check_container_deviation(
            working_df, labels_, cluster_profiles, myInstance.dict_id_c)
        plot.update_containers_plot(fig_cont, ax_cont, working_df, tmax)
        plot.update_nodes_plot(fig_cont, ax_cont, working_df, tmax)
        tmin += 1
        tmax += 1
        if tmax >= myInstance.time:
            end = True

    print('Total computing time : %fs' % (time.time() - main_time))

    plt.show()


if __name__ == '__main__':
    main()
