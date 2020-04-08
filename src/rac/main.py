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
    my_instance = instance.Instance(data)

    building_cplex_time = time.time()
    cplex_model = model.CPXInstance(my_instance)
    print('CPLEX model imported.')
    print('Building CPLEX model time : %fs' %
          (time.time() - building_cplex_time))
    # TODO fix this method
    # print('Objective of initial placement : ',
    #       cplex_model.get_obj_value_heuristic(my_instance))

    # Plot data (containers & nodes consumption)
    ctnr.plot_all_data_all_containers(
        my_instance.df_containers, sep_time=my_instance.sep_time)
    plot.plot_containers_groupby_nodes(
        my_instance.df_containers,
        my_instance.df_nodes_meta.cpu.max(),
        my_instance.sep_time)

    plt.show(block=False)
    input('Press any key to continue ...')

    #####################################################################
    # Clustering part
    df_containers_clust = clt.build_matrix_indiv_attr(
        my_instance.df_containers.loc[
            my_instance.df_containers['timestamp'] <= my_instance.sep_time])

    clustering_time = time.time()

    labels_ = clt.perform_clustering(
        df_containers_clust, clustering_algo, my_instance.nb_clusters)
    df_containers_clust['cluster'] = labels_
    my_instance.nb_clusters = labels_.max() + 1

    # print('Clustering balance : ',
    #       df_containers_clust.groupby('cluster').count())

    # TODO improve this part (distance...)
    cluster_vars = clt.get_cluster_variance(
        my_instance.nb_clusters, df_containers_clust)
    cluster_profiles = clt.get_cluster_mean_profile(
        my_instance.nb_clusters,
        df_containers_clust,
        my_instance.window_duration)
    cluster_var_matrix = clt.get_sum_cluster_variance(
        cluster_profiles, cluster_vars)
    # cluster_distance = clt.get_distance_cluster(
    #     my_instance, cluster_profiles)

    print('Clustering computing time : %fs' % (time.time() - clustering_time))

    #####################################################################
    # Plot clustering
    do_plot_clustering = True
    if do_plot_clustering:
        # plot.plot_containers_clustering_together(df_containers_clust)
        plot.plot_clustering(df_containers_clust)
        plot.plot_cluster_profiles(cluster_profiles)

        # plot.plot_clustering_contaiers_by_node(
        #     my_instance, labels_)

    #####################################################################
    # Allocation
    allocation_time = time.time()
    containers_grouped = alloc.allocation_distant_pairwise(
        my_instance, cluster_var_matrix, labels_)

    # plot.plot_containers_groupby_nodes(my_instance.df_containers)

    # alloc.allocation_ffd(my_instance, cluster_vars, cluster_var_matrix,
    #                      labels_)

    print('Allocation time : %fs' % (time.time() - allocation_time))

    # TODO make comparison between heuristics

    # Plot node's data
    # nd.plot_all_data_all_nodes(df_nodes)
    plot.plot_containers_groupby_nodes(
        my_instance.df_containers,
        my_instance.df_nodes_meta.cpu.max(),
        my_instance.sep_time)
    # nd.plot_all_data_all_nodes_end(my_instance.df_nodes, my_instance.time)

    # Check if sum(nodes.conso) for all tick is same as previous
    # (global_max_cpu, global_max_mem) = nd.plot_total_usage(
    #     my_instance.df_nodes, 'Total conso on all nodes after new alloc')
    # print(global_max_cpu, global_max_mem)

    # nd.print_vmr(my_instance.df_nodes, my_instance.time, 2)
    # my_instance.instance_in_file_after(instance_file)
    # my_instance.nodes_state_inFile(instance_file)
    # nd.get_mean_consumption(my_instance.df_nodes)
    # nd.get_variance_consumption(my_instance.df_nodes)

    # plt.show(block=False)
    plt.show(block=True)

    #####################################################################
    # CPLEX evaluation part
    # Update the solution in CPLEX model
    cplex_model.set_x_from_df(my_instance)
    # cplex_model.get_obj_value_heuristic(my_instance)
    # cplex_model.print_sol_infos_heur()
    print('Adding constraints from heuristic ...')
    cplex_model.add_constraint_heuristic(containers_grouped, my_instance)

    print('Solving linear relaxation ...')
    # cplex_model.solve_relax()
    # model.print_all_dual(cplex_model.relax_mdl)
    # cplex_model.get_max_dual()

    #####################################################################
    # Evaluation period
    fig_cont, ax_cont = plot.init_containers_plot(
        my_instance.df_containers, my_instance.sep_time)
    fig_node, ax_node = plot.init_nodes_plot(
        my_instance.df_containers, my_instance.sep_time)
    fig_clust, ax_clust = plot.init_plot_clustering(df_containers_clust)

    tmin = my_instance.df_containers['timestamp'].min()
    tmax = my_instance.sep_time

    end = False

    while not end:
        working_df = my_instance.df_containers[
            (my_instance.
             df_containers['timestamp'] >= tmin) & (my_instance.
                                                    df_containers['timestamp'] <= tmax)]
        df_clust = clt.build_matrix_indiv_attr(working_df)
        df_clust['cluster'] = labels_
        cluster_profiles = clt.get_cluster_mean_profile(
            my_instance.nb_clusters, df_clust,
            my_instance.window_duration, tmin)
        plot.update_clustering_plot(fig_clust, ax_clust, df_clust)
        clt.check_container_deviation(
            working_df, labels_, cluster_profiles, my_instance.dict_id_c)
        plot.update_containers_plot(fig_cont, ax_cont, working_df, tmax)
        plot.update_nodes_plot(fig_cont, ax_cont, working_df, tmax)
        tmin += 1
        tmax += 1
        if tmax >= my_instance.time:
            end = True

    print('Total computing time : %fs' % (time.time() - main_time))

    plt.show()


if __name__ == '__main__':
    main()
