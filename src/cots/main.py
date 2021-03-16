"""
=========
cots main
=========
Entry point of cots module through ``cots --data --params``.

    - data is the folder where we find the files
        x container_usage.csv : describes container resource consumption
        x node_meta.csv : describes nodes capacities
        (x node_usage.csv : describes nodes resource consumption)

The entire methodology is called from here (initialization, clustering,
allocation, evaluation, access to optimization model...).
"""

import logging
import time
from typing import List

import click

from matplotlib import pyplot as plt

import pandas as pd

# Personnal imports
from . import allocation as alloc
from . import clustering as clt
from . import container as ctnr
from . import init as it
# from . import model
# TODO set in params
# from . import model_cplex as mc
from . import model_cplex_clustering as mc
from . import placement as place
from . import plot
from .instance import Instance


# Clustering algorithm
# {kmeans, hierarchical, spectral, spectral_perso}

# TODO add 'help' message
@click.command()
# @click.option('--data', required=True, type=click.Path(exists=True))
@click.option('--params', required=True, type=click.Path(exists=True))
def main(params):
    """Perform all things of methodology."""
    # Initialization part
    main_time = time.time()

    config = it.read_params(params)
    logging.basicConfig(filename=config['data']['path'] + '/logs.log', filemode='w',
                        format='%(message)s', level=logging.INFO)
    plt.style.use('bmh')

    # Init containers & nodes data, then Instance
    logging.info('Loading data and creating Instance (Instance information are in results file)\n')
    my_instance = Instance(config)

    # Use pyomo model => to be fully applied after tests
    # my_model = model.create_model(config['optimization']['model'], my_instance)
    # model.solve_model(my_model, 'glpk')
    # input('waiting here')

    # Plot initial data (containers & nodes consumption)
    ctnr.plot_all_data_all_containers(
        my_instance.df_indiv, sep_time=my_instance.sep_time)

    # plot.plot_containers_groupby_nodes(
    #     my_instance.df_indiv,
    #     my_instance.df_host_meta[it.metrics[0]].max(),
    #     my_instance.sep_time)

    # Print real objective value of second part if no loop
    # logging.info('Real objective value of second part without heuristic and loop')
    # mc.get_obj_value_heuristic(my_instance.df_indiv,
    #                            my_instance.sep_time,
    #                            my_instance.df_indiv[it.tick_field].max())

    # Get dataframe of current part
    working_df_indiv = my_instance.df_indiv.loc[
        my_instance.df_indiv[it.tick_field] <= my_instance.sep_time
    ]

    # Clustering part
    logging.info('Starting first clustering ...')
    (df_indiv_clust, my_instance.dict_id_c) = clt.build_matrix_indiv_attr(
        working_df_indiv)

    clustering_time = time.time()
    labels_ = clt.perform_clustering(
        df_indiv_clust, config['clustering']['algo'], my_instance.nb_clusters)
    df_indiv_clust['cluster'] = labels_
    my_instance.nb_clusters = labels_.max() + 1

    # TODO improve this part (distance...)
    cluster_vars = clt.get_cluster_variance(
        my_instance.nb_clusters, df_indiv_clust)
    cluster_profiles = clt.get_cluster_mean_profile(
        my_instance.nb_clusters,
        df_indiv_clust,
        working_df_indiv[it.tick_field].nunique())
    cluster_var_matrix = clt.get_sum_cluster_variance(
        cluster_profiles, cluster_vars)

    it.results_file.write('\nClustering computing time : %fs\n\n' % (time.time() - clustering_time))

    # Placement
    containers_grouped = []
    if config['placement']['enable']:
        logging.info('Performing placement ... \n')
        placement_time = time.time()
        containers_grouped = place.allocation_distant_pairwise(
            my_instance, cluster_var_matrix, labels_)

        it.results_file.write('\nPlacement time : %fs\n\n' % (time.time() - placement_time))
    else:
        logging.info('We do not perform placement \n')

    # Plot clustering & allocation for 1st part
    plot_before_loop = False
    if plot_before_loop:
        spec_containers = False
        if spec_containers:
            ctnr.show_specific_containers(working_df_indiv, df_indiv_clust,
                                          my_instance, labels_)
        show_clustering = False
        if show_clustering:
            # plot.plot_clustering_containers_by_node(
            #     working_df_indiv, my_instance.dict_id_c, labels_)
            plot.plot_clustering(df_indiv_clust, my_instance.dict_id_c,
                                 title='Clustering on first half part')
        # plot.plot_containers_groupby_nodes(
        #     my_instance.df_indiv,
        #     my_instance.df_host_meta.cpu.max(),
        #     my_instance.sep_time)

    # Print real objective value
    # mc.get_obj_value_heuristic(my_instance.df_indiv,
    #                            working_df_indiv[it.tick_field].min(),
    #                            working_df_indiv[it.tick_field].max())

    # Plot heuristic result without loop
    plot.plot_containers_groupby_nodes(
        my_instance.df_indiv,
        my_instance.df_host_meta[it.metrics[0]].max(),
        my_instance.sep_time,
        title='Node consumption after heuristic without loop')

    # Print real objective value of second part if no loop
    print('Real objective value of second part without loop')
    mc.get_obj_value_heuristic(my_instance.df_indiv,
                               my_instance.sep_time,
                               my_instance.df_indiv[it.tick_field].max())

    # input('\nEnd of first part, press enter to enter loop ...\n')

    # Test allocation use case
    # TODO specific window not taken into account
    if config['allocation']['enable']:
        logging.info('Performing allocation ... \n')
        print(alloc.check_constraints(
            my_instance, working_df_indiv, config['allocation']))
    else:
        logging.info('We do not perform allocation \n')

    # Plot heuristic result without loop
    # ctnr.plot_all_data_all_containers(
    #     my_instance.df_indiv, sep_time=my_instance.sep_time)
    # plot.plot_containers_groupby_nodes(
    #     my_instance.df_indiv,
    #     my_instance.df_host_meta[it.metrics[0]].max(),
    #     my_instance.sep_time,
    #     title='Node consumption after heuristic and change allocation')

    # plt.show(block=False)

    # loop 'streaming' progress
    streaming_eval(my_instance, df_indiv_clust, labels_,
                   containers_grouped, config['loop']['tick'],
                   config['loop']['constraints_dual'],
                   config['loop']['tol_dual_clust'],
                   config['loop']['tol_dual_place'])

    # Plot after the loop
    plot.plot_containers_groupby_nodes(
        my_instance.df_indiv,
        my_instance.df_host_meta[it.metrics[0]].max(),
        my_instance.sep_time,
        title='Node consumption with loop')

    # Print real objective value
    print('Real objective value of second part with loop')
    mc.get_obj_value_heuristic(my_instance.df_indiv,
                               my_instance.sep_time,
                               my_instance.df_indiv[it.tick_field].max())

    it.results_file.write('\nTotal computing time : %fs' % (time.time() - main_time))

    it.results_file.close()


def streaming_eval(my_instance: Instance, df_indiv_clust: pd.DataFrame,
                   labels_: List, containers_grouped: List, tick: int,
                   constraints_dual: List,
                   tol_clust: float, tol_place: float):
    """Define the streaming process for evaluation."""
    fig_node, ax_node = plot.init_nodes_plot(
        my_instance.df_indiv, my_instance.sep_time,
        my_instance.df_host_meta[it.metrics[0]].max()
    )
    fig_clust, ax_clust = plot.init_plot_clustering(
        df_indiv_clust, my_instance.dict_id_c)

    cluster_profiles = clt.get_cluster_mean_profile(
        my_instance.nb_clusters,
        df_indiv_clust,
        my_instance.window_duration)
    fig_mean_clust, ax_mean_clust = plot.init_plot_cluster_profiles(
        cluster_profiles
    )

    tmin = my_instance.df_indiv[it.tick_field].min()
    tmax = my_instance.sep_time
    current_obj_func = 1
    # current_nb_nodes = my_instance.nb_nodes
    # max_dual_clust = None
    clustering_dual_values = {}
    # placement_dual_values = {}

    loop_nb = 1
    end = False

    logging.info('Beginning the loop process ...\n')
    while not end:
        logging.info('Enter loop number %d\n' % loop_nb)
        it.results_file.write('Loop number %d\n' % loop_nb)

        working_df_indiv = my_instance.df_indiv[
            (my_instance.
                df_indiv[it.tick_field] >= tmin) & (
                my_instance.df_indiv[it.tick_field] <= tmax)]
        (df_clust, my_instance.dict_id_c) = clt.build_matrix_indiv_attr(
            working_df_indiv)
        w = clt.build_similarity_matrix(df_clust)
        df_clust['cluster'] = labels_
        u = clt.build_adjacency_matrix(labels_)
        # v = place.build_placement_adj_matrix(
        #     working_df_indiv, my_instance.dict_id_c)

        # TODO not very practical
        # plot.plot_clustering_containers_by_node(
        #     working_df_indiv, my_instance.dict_id_c, labels_)

        # evaluate clustering solution from optim model
        # TODO update model from existing one, not creating new one each time
        cplex_model = mc.CPXInstance(working_df_indiv,
                                     my_instance.df_host_meta,
                                     my_instance.nb_clusters,
                                     my_instance.dict_id_c,
                                     my_instance.dict_id_n,
                                     obj_func=current_obj_func,
                                     w=w, u=u, pb_number=2)
        logging.info('# Clustering evaluation #')
        logging.info('Solving linear relaxation ...')
        cplex_model.solve(cplex_model.relax_mdl)
        # TODO only mustlink constraints
        # mc.print_all_dual(cplex_model.relax_mdl,
        #                   nn_only=True, names=constraints_dual)

        if len(clustering_dual_values) == 0:
            logging.info('Clustering problem not evaluated yet\n')
            clustering_dual_values = mc.fill_constraints_dual_values(
                cplex_model.relax_mdl, constraints_dual
            )
        else:
            logging.info('Checking for changes in clustering dual values ...')
            moving_containers = mc.get_moving_containers_clust(
                cplex_model.relax_mdl, clustering_dual_values,
                tol_clust, my_instance.nb_containers, my_instance.dict_id_c,
                df_clust, cluster_profiles)
            if len(moving_containers) > 0:
                logging.info('Changing clustering ...')
                clt.change_clustering(moving_containers, df_clust, cluster_profiles)
            else:
                logging.info('Clustering seems still right ...')
                it.results_file.write('Clustering seems still right ...')

        # if max_dual_clust is None:
        #     print('We have not evaluated the model yet...\n')
        #     max_dual_clust = mc.get_max_dual(
        #         cplex_model.relax_mdl, constraints_dual)
        # else:
        #     if mc.dual_changed(cplex_model.relax_mdl, constraints_dual,
        #                        max_dual_clust, tol_clust):
        #         print('Performing new clustering ...\n')
        #         labels_ = clt.perform_clustering(
        #             df_clust, 'kmeans', my_instance.nb_clusters)
        #         df_clust['cluster'] = labels_
        #         my_instance.nb_clusters = labels_.max() + 1
        #         u = clt.build_adjacency_matrix(labels_)
        #         cplex_model = mc.CPXInstance(working_df_indiv,
        #                                      my_instance.df_host_meta,
        #                                      my_instance.nb_clusters,
        #                                      my_instance.dict_id_c,
        #                                      my_instance.dict_id_n,
        #                                      obj_func=current_obj_func,
        #                                      w=w, u=u, pb_number=2)
        #         print('\n')
        #         print('Solving linear relaxation ...')
        #         cplex_model.solve(cplex_model.relax_mdl)
        #     else:
        #         print('Clustering seems still right\n')
        #     max_dual_clust = mc.get_max_dual(
        #         cplex_model.relax_mdl, constraints_dual)

        # Compute new clusters profiles
        cluster_profiles = clt.get_cluster_mean_profile(
            my_instance.nb_clusters,
            df_clust,
            my_instance.window_duration, tmin=tmin)

        # TODO improve this part (use cluster profiles)
        # dv = ctnr.build_var_delta_matrix(
        #     working_df_indiv, my_instance.dict_id_c)

        # TODO Evaluate this possibility
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
        #         containers_grouped = place.allocation_distant_pairwise(
        #             my_instance, cluster_var_matrix, labels_,
        #             cplex_model.max_open_nodes)
        #     else:
        #         cplex_model.update_obj_function(1)
        #         current_obj_func += 1

        # evaluate placement
        # cplex_model = mc.CPXInstance(working_df_indiv,
        #                              my_instance.df_host_meta,
        #                              my_instance.nb_clusters,
        #                              my_instance.dict_id_c,
        #                              my_instance.dict_id_n,
        #                              obj_func=current_obj_func,
        #                              w=w, u=u, v=v, dv=dv, pb_number=3)

        # print('Adding constraints from heuristic ...\n')
        # cplex_model.add_constraint_heuristic(
        #     containers_grouped, my_instance)
        # print('Solving linear relaxation ...')
        # cplex_model.solve(cplex_model.relax_mdl)
        # mc.print_all_dual(cplex_model.relax_mdl,
        #                   nn_only=True, names=constraints_dual)

        # moving_containers = []
        # if len(placement_dual_values) == 0:
        #     print('Placement problem not evaluated yet\n')
        #     placement_dual_values = mc.fill_constraints_dual_values(
        #         cplex_model.relax_mdl, constraints_dual
        #     )
        # else:
        #     print('Checking for changes in placement dual values ...')
        #     print(placement_dual_values)
        #     moving_containers = mc.get_moving_containers(
        #         cplex_model.relax_mdl, placement_dual_values,
        #         tol_place, my_instance.nb_containers)
        #     print(moving_containers)

        # Move containers by hand
        # print('Enter the containers you want to move')
        # moving_containers = []
        # while True:
        #     moving_container = input(
        #         'Enter a container you want to move, or press Enter')
        #     if moving_container.isdigit():
        #         moving_containers.append(int(moving_container))
        #         for pair in containers_grouped:
        #             if my_instance.dict_id_c[int(moving_container)] in pair:
        #                 containers_grouped.remove(pair)
        #     else:
        #         print('End of input.')
        #         break
        # if len(moving_containers) >= 1:
        #     place.move_list_containers(moving_containers, my_instance,
        #                                working_df_indiv[it.tick_field].min(),
        #                                working_df_indiv[it.tick_field].max())

        # else:
        #     print('No container to move : we do nothing ...\n')

        # update clustering & node consumption plot
        plot.update_clustering_plot(
            fig_clust, ax_clust, df_clust, my_instance.dict_id_c)
        plot.update_cluster_profiles(fig_mean_clust, ax_mean_clust, cluster_profiles,
                                     sorted(working_df_indiv[it.tick_field].unique()))
        plot.update_nodes_plot(fig_node, ax_node, working_df_indiv)
        # plt.show(block=False)

        # input('\nPress any key to progress in time ...\n')
        tmin += tick
        tmax += tick
        loop_nb += 1
        if tmax >= my_instance.time:
            end = True


if __name__ == '__main__':
    main()
