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
from typing import List

import click

from matplotlib import pyplot as plt

import pandas as pd

# Personnal imports
from . import allocation as alloc
from . import clustering as clt
from . import container as ctnr
from . import model
from . import plot
from .instance import Instance


# Clustering algorithm
# {kmeans, hierarchical, spectral, spectral_perso}
# TODO as option ? + nb_clusters
clustering_algo = 'hierarchical'


@click.command()
@click.argument('data', type=click.Path(exists=True))
def main(data):
    """Perform all things of methodology."""
    # Initialization part
    main_time = time.time()

    # Init containers & nodes data, then Instance
    my_instance = Instance(data)

    # Plot initial data (containers & nodes consumption)
    ctnr.plot_all_data_all_containers(
        my_instance.df_containers, sep_time=my_instance.sep_time)
    plot.plot_containers_groupby_nodes(
        my_instance.df_containers,
        my_instance.df_nodes_meta.cpu.max(),
        my_instance.sep_time)

    plt.show(block=False)
    input('Press enter to continue ...')

    # Get dataframe of current part
    working_df_containers = my_instance.df_containers.loc[
        my_instance.df_containers['timestamp'] <= my_instance.sep_time
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

    # Clustering part
    (df_containers_clust, my_instance.dict_id_c) = clt.build_matrix_indiv_attr(
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

    print('Clustering computing time : %fs' % (time.time() - clustering_time))

    # Allocation
    allocation_time = time.time()
    containers_grouped = alloc.allocation_distant_pairwise(
        my_instance, cluster_var_matrix, labels_)

    # alloc.allocation_ffd(my_instance, cluster_vars, cluster_var_matrix,
    #                      labels_)

    print('Allocation time : %fs' % (time.time() - allocation_time))

    working_df_containers = my_instance.df_containers.loc[
        my_instance.df_containers['timestamp'] <= my_instance.sep_time
    ]

    # CPLEX evaluation part
    cplex_model = model.CPXInstance(working_df_containers,
                                    my_instance.df_nodes_meta,
                                    my_instance.dict_id_c,
                                    my_instance.dict_id_n,
                                    obj_func=1)
    cplex_model.set_x_from_df(my_instance.df_containers,
                              my_instance.dict_id_c,
                              my_instance.dict_id_n)
    print('Adding constraints from heuristic ...')
    cplex_model.add_constraint_heuristic(containers_grouped, my_instance)

    print('Solving linear relaxation ...')
    cplex_model.solve_relax()
    model.print_all_dual(cplex_model.relax_mdl, True)
    cplex_model.get_obj_value_heuristic()
    # cplex_model.get_max_dual()

    # Plot clustering & allocation for 1st part
    plot_before_loop = False
    if plot_before_loop:
        spec_containers = False
        if spec_containers:
            print('Enter list of containers to show separated by a comma :')
            containers_input = input()
            containers_toshow = [int(s) for s in containers_input.split(',')]
            plot.plot_clustering_containers_by_node_spec_cont(
                working_df_containers, my_instance.dict_id_c,
                labels_, containers_toshow
            )
            plot.plot_clustering_spec_cont(df_containers_clust,
                                           my_instance.dict_id_c,
                                           containers_toshow,
                                           title='Clustering on first half part')
        else:
            plot.plot_clustering_containers_by_node(
                working_df_containers, my_instance.dict_id_c, labels_)
            plot.plot_clustering(df_containers_clust, my_instance.dict_id_c,
                                 title='Clustering on first half part')
        plot.plot_containers_groupby_nodes(
            my_instance.df_containers,
            my_instance.df_nodes_meta.cpu.max(),
            my_instance.sep_time)

    plt.show(block=True)

    # loop at same time or 'streaming' progress
    streaming = True
    if streaming:
        streaming_eval(my_instance, df_containers_clust, labels_, containers_grouped)
    else:
        # loop to test alloc changes for same time
        while True:
            print('Enter the container you want to move')
            moving_container = input()

            if moving_container.isdigit():
                alloc.move_container(int(moving_container), working_df_containers,
                                     my_instance)
                for pair in containers_grouped:
                    if my_instance.dict_id_c[int(moving_container)] in pair:
                        containers_grouped.remove(pair)
            else:
                print('Not int : we do nothing ...')

            # working_df_containers = my_instance.df_containers.loc[
            #     (my_instance.df_containers['timestamp'] >= 4) & (
            #         my_instance.df_containers['timestamp'] <= 10
            #     )
            # ]
            working_df_containers = my_instance.df_containers.loc[
                my_instance.df_containers['timestamp'] <= my_instance.sep_time
            ]

            plot.plot_clustering_containers_by_node(
                working_df_containers, my_instance.dict_id_c, labels_)
            plt.show(block=False)
            cplex_model = model.CPXInstance(working_df_containers,
                                            my_instance.df_nodes_meta,
                                            my_instance.dict_id_c,
                                            my_instance.dict_id_n,
                                            obj_func=1)
            cplex_model.set_x_from_df(working_df_containers,
                                      my_instance.dict_id_c,
                                      my_instance.dict_id_n)
            print('Adding constraints from heuristic ...')
            cplex_model.add_constraint_heuristic(containers_grouped, my_instance)
            cplex_model.solve_relax()
            model.print_all_dual(cplex_model.relax_mdl, True)
            cplex_model.get_obj_value_heuristic()

    print('Total computing time : %fs' % (time.time() - main_time))

    plt.show()


def streaming_eval(my_instance: Instance, df_containers_clust: pd.DataFrame,
                   labels_: List, containers_grouped: List):
    """Define the streaming process for evaluation."""
    fig_node, ax_node = plot.init_nodes_plot(
        my_instance.df_containers, my_instance.sep_time,
        my_instance.df_nodes_meta.cpu.max())
    fig_clust, ax_clust = plot.init_plot_clustering(
        df_containers_clust, my_instance.dict_id_c)

    tmin = my_instance.df_containers['timestamp'].min()
    tmax = my_instance.sep_time
    current_obj_func = 1
    # current_nb_nodes = my_instance.nb_nodes

    end = False

    while not end:
        working_df_containers = my_instance.df_containers[
            (my_instance.
                df_containers['timestamp'] >= tmin) & (
                my_instance.df_containers['timestamp'] <= tmax)]
        (df_clust, my_instance.dict_id_c) = clt.build_matrix_indiv_attr(
            working_df_containers)
        df_clust['cluster'] = labels_
        plot.update_clustering_plot(
            fig_clust, ax_clust, df_clust, my_instance.dict_id_c)
        # clt.check_container_deviation(
        #     working_df_containers, labels_,
        # cluster_profiles, my_instance.dict_id_c)

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
        print('Adding constraints from heuristic ...')
        cplex_model.add_constraint_heuristic(
            containers_grouped, my_instance)
        print('Solving linear relaxation ...')
        cplex_model.solve_relax()
        model.print_all_dual(cplex_model.relax_mdl, True)
        cplex_model.get_obj_value_heuristic()

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
        print('Enter the container you want to move')
        moving_container = input()
        if moving_container.isdigit():
            alloc.move_container(int(moving_container), working_df_containers,
                                 my_instance)
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


if __name__ == '__main__':
    main()
