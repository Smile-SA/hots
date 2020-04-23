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
# from . import container as ctnr
from . import model
from . import plot
from .instance import Instance


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
    my_instance = Instance(data)

    # Plot data (containers & nodes consumption)
    # ctnr.plot_all_data_all_containers(
    #     my_instance.df_containers, sep_time=my_instance.sep_time)
    # plot.plot_containers_groupby_nodes(
    #     my_instance.df_containers,
    #     my_instance.df_nodes_meta.cpu.max(),
    #     my_instance.sep_time)

    # plt.show(block=False)
    # input('Press any key to continue ...')

    # Get dataframe of current part
    working_df_containers = my_instance.df_containers.loc[
        my_instance.df_containers['timestamp'] < my_instance.sep_time
    ]

    # Build optimization model (only if we want initial obj value)
    # building_cplex_time = time.time()
    # cplex_model = model.CPXInstance(working_df_containers,
    #                                 my_instance.df_nodes_meta,
    #                                 my_instance.dict_id_c,
    #                                 my_instance.dict_id_n, obj_func=1)
    # print('CPLEX model imported.')
    # print('Building CPLEX model time : %fs' %
    #       (time.time() - building_cplex_time))
    # cplex_model.get_obj_value_heuristic()
    # cplex_model.solve_relax()
    # model.print_all_dual(cplex_model.relax_mdl, True)

    #####################################################################
    # Clustering part
    df_containers_clust = clt.build_matrix_indiv_attr(
        working_df_containers)

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
        working_df_containers['timestamp'].nunique())
    cluster_var_matrix = clt.get_sum_cluster_variance(
        cluster_profiles, cluster_vars)
    # cluster_distance = clt.get_distance_cluster(
    #     my_instance, cluster_profiles)

    print('Clustering computing time : %fs' % (time.time() - clustering_time))

    #####################################################################
    # Plot clustering
    # TODO add in loop
    # do_plot_clustering = True
    # if do_plot_clustering:
    # plot.plot_containers_clustering_together(df_containers_clust)
    # plot.plot_clustering(df_containers_clust,
    #                      title='Clustering on first half part')
    # plot.plot_cluster_profiles(cluster_profiles)

    # plot.plot_clustering_contaiers_by_node(
    #     my_instance, labels_)

    #####################################################################
    # Allocation
    allocation_time = time.time()
    nb_open_nodes = 4
    containers_grouped = alloc.allocation_distant_pairwise(
        my_instance, cluster_var_matrix, labels_, nb_open_nodes)

    # plot.plot_containers_groupby_nodes(my_instance.df_containers)

    # alloc.allocation_ffd(my_instance, cluster_vars, cluster_var_matrix,
    #                      labels_)

    print('Allocation time : %fs' % (time.time() - allocation_time))
    # working_df_containers = my_instance.df_containers.loc[
    #     my_instance.df_containers['timestamp'] < my_instance.sep_time
    # ]
    # cplex_model = model.CPXInstance(working_df_containers,
    #                                 my_instance.df_nodes_meta,
    #                                 my_instance.dict_id_c,
    #                                 my_instance.dict_id_n,
    #                                 obj_func=1)

    # TODO make comparison between heuristics

    # Plot node's data
    # nd.plot_all_data_all_nodes(df_nodes)
    # plot.plot_containers_groupby_nodes(
    #     my_instance.df_containers,
    #     my_instance.df_nodes_meta.cpu.max(),
    #     my_instance.sep_time)
    # nd.plot_all_data_all_nodes_end(my_instance.df_nodes, my_instance.time)

    #####################################################################
    # CPLEX evaluation part
    # Update the solution in CPLEX model
    # cplex_model.set_x_from_df(my_instance.df_containers,
    #                           my_instance.dict_id_c,
    #                           my_instance.dict_id_n)
    # cplex_model.get_obj_value_heuristic(my_instance.df_nodes,
    #                                     my_instance.dict_id_n)
    # cplex_model.print_sol_infos_heur()
    # print('Adding constraints from heuristic ...')
    # cplex_model.add_constraint_heuristic(containers_grouped, my_instance)

    # print('Solving linear relaxation ...')
    # cplex_model.solve_relax()
    # model.print_all_dual(cplex_model.relax_mdl, True)
    # cplex_model.get_obj_value_heuristic()
    # cplex_model.get_max_dual()

    # plt.show(block=False)
    # plt.show(block=False)

    # containers_grouped = alloc.allocation_distant_pairwise(
    #     my_instance, cluster_var_matrix, labels_, cplex_model.max_open_nodes - 1)

    # plot.plot_containers_groupby_nodes(
    #     my_instance.df_containers,
    #     my_instance.df_nodes_meta.cpu.max(),
    #     my_instance.sep_time)
    # cplex_model.set_x_from_df(my_instance)
    # cplex_model.get_obj_value_heuristic(my_instance)

    # plt.show(block=False)
    # input('Press any key to begin loop ...')

    # loop to test alloc changes for same time
    # while True:
    #     print('Enter the container you want to move')
    #     moving_container = input()
    #     working_df_containers = my_instance.df_containers.loc[
    #         my_instance.df_containers['timestamp'] < my_instance.sep_time
    #     ]
    #     alloc.move_container(int(moving_container), working_df_containers,
    #                          my_instance, nb_open_nodes)

    #     for pair in containers_grouped:
    #         if my_instance.dict_id_c[int(moving_container)] in pair:
    #             containers_grouped.remove(pair)

    #     plot.plot_clustering_containers_by_node(my_instance, labels_)
    #     plt.show(block=False)
    #     cplex_model = model.CPXInstance(working_df_containers,
    #                                     my_instance.df_nodes_meta,
    #                                     my_instance.dict_id_c,
    #                                     my_instance.dict_id_n,
    #                                     obj_func=1)
    #     cplex_model.set_x_from_df(working_df_containers,
    #                               my_instance.dict_id_c,
    #                               my_instance.dict_id_n)
    #     print('Adding constraints from heuristic ...')
    #     cplex_model.add_constraint_heuristic(containers_grouped, my_instance)
    #     cplex_model.solve_relax()
    #     model.print_all_dual(cplex_model.relax_mdl, True)
    #     cplex_model.get_obj_value_heuristic()
    # input()

    #####################################################################
    # Evaluation period
    # fig_cont, ax_cont = plot.init_containers_plot(
    #     my_instance.df_containers, my_instance.sep_time)
    fig_node, ax_node = plot.init_nodes_plot(
        my_instance.df_containers, my_instance.sep_time)
    fig_clust, ax_clust = plot.init_plot_clustering(df_containers_clust)

    tmin = my_instance.df_containers['timestamp'].min()
    tmax = my_instance.sep_time
    current_obj_func = 1
    # current_nb_nodes = my_instance.nb_nodes

    end = False

    while not end:
        # update clustering plot
        working_df_containers = my_instance.df_containers[
            (my_instance.
             df_containers['timestamp'] >= tmin) & (
                 my_instance.df_containers['timestamp'] <= tmax)]
        df_clust = clt.build_matrix_indiv_attr(working_df_containers)
        df_clust['cluster'] = labels_
        # cluster_profiles = clt.get_cluster_mean_profile(
        #     my_instance.nb_clusters, df_clust,
        #     my_instance.window_duration, tmin)
        plot.update_clustering_plot(fig_clust, ax_clust, df_clust)
        # clt.check_container_deviation(
        #     working_df_containers, labels_,
        # cluster_profiles, my_instance.dict_id_c)

        # plot.update_containers_plot(
        # fig_cont, ax_cont, working_df_containers, tmax)

        # evaluate solution from optim model
        # TODO update model from existing one, not creating new one each time
        cplex_model = model.CPXInstance(working_df_containers,
                                        my_instance.df_nodes_meta,
                                        my_instance.dict_id_c,
                                        my_instance.dict_id_n,
                                        obj_func=current_obj_func)
        # nb_nodes=current_nb_nodes)
        cplex_model.set_x_from_df(working_df_containers,
                                  my_instance.dict_id_c,
                                  my_instance.dict_id_n)
        cplex_model.get_obj_value_heuristic()
        print('Adding constraints from heuristic ...')
        cplex_model.add_constraint_heuristic(containers_grouped, my_instance)
        print('Solving linear relaxation ...')
        cplex_model.solve_relax()
        model.print_all_dual(cplex_model.relax_mdl, True)

        # if not cplex_model.obj_func:
        #     if cplex_model.relax_mdl.get_constraint_by_name(
        #             'max_nodes').dual_value > 0:
        #         print(cplex_model.relax_mdl.get_constraint_by_name(
        #             'max_nodes').to_string())
        #         print(cplex_model.relax_mdl.get_constraint_by_name(
        #             'max_nodes').dual_value)
        #         current_nb_nodes = current_nb_nodes - 1
        #         cplex_model.update_max_nodes_ct(current_nb_nodes)

        #         # TODO adapt existing solution, but not from scratch
        #         containers_grouped = alloc.allocation_distant_pairwise(
        #             my_instance, cluster_var_matrix, labels_,
        #             cplex_model.max_open_nodes)
        #     else:
        #         cplex_model.update_obj_function(1)
        #         current_obj_func += 1

        # update node consumption plot
        plot.update_nodes_plot(fig_node, ax_node, working_df_containers, tmax)
        plot.plot_clustering_containers_by_node(
            working_df_containers, my_instance.dict_id_c, labels_)
        plt.show(block=False)
        print(containers_grouped)
        print(my_instance.dict_id_c)
        print('Enter the container you want to move')
        moving_container = input()
        if moving_container.isdigit():
            alloc.move_container(int(moving_container), working_df_containers,
                                 my_instance, nb_open_nodes)
            for pair in containers_grouped:
                if my_instance.dict_id_c[int(moving_container)] in pair:
                    containers_grouped.remove(pair)
        else:
            print('Not int : we do nothing ...')

        input('Press any key to progress in time ...')
        tmin += 1
        tmax += 1
        if tmax >= my_instance.time:
            end = True

    print('Total computing time : %fs' % (time.time() - main_time))

    plt.show()


if __name__ == '__main__':
    main()
