# coding=utf-8
"""
=========
cots main
=========
Entry point of cots module through ``cots --path [--k --tau]``.

    - path is the folder where we find the files
        x container_usage.csv : describes container resource consumption
        x node_meta.csv : describes nodes capacities
        (x node_usage.csv : describes nodes resource consumption)
    - k is the number of cluster to use for the clustering part
    - tau is the size of one time window (for analysis and solution evaluation)

The entire methodology is called from here (initialization, clustering,
allocation, evaluation, access to optimization model...).
"""
# print(__doc__)

import logging
import time
from typing import Dict, List

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

# Personnal imports
# from . import allocation as alloc
from . import clustering as clt
from . import container as ctnr
from . import init as it
# from . import model
# TODO set in params
# from . import model_cplex as mc
from . import model_cplex_clustering as mc
from . import node
from . import placement as place
from . import plot
from .instance import Instance


@click.command()
@click.argument('path', required=True, type=click.Path(exists=True))
@click.option('-k', required=False, type=int, help='Number of clusters')
@click.option('-t', '--tau', required=False, type=int, help='Time window size')
@click.option('-m', '--method', required=False, type=str, default='loop', help='Method used')
def main(path, k, tau, method):
    """Use method to propose a placement solution for micro-services adjusted in time."""
    # Initialization part
    main_time = time.time()

    if not path[-1] == '/':
        path += '/'

    # TODO what if we want tick < tau ?
    (config, output_path) = it.read_params(path, k, tau, method)
    logging.basicConfig(filename=output_path + '/logs.log', filemode='w',
                        format='%(message)s', level=logging.INFO)
    plt.style.use('bmh')

    # Init containers & nodes data, then Instance
    logging.info('Loading data and creating Instance (Instance information are in results file)\n')
    my_instance = Instance(path, config)
    it.results_file.write('Method used : %s\n' % method)

    # Use pyomo model => to be fully applied after tests
    # my_model = model.create_model(config['optimization']['model'], my_instance)
    # model.solve_model(my_model, 'glpk')

    # Plot initial data (containers)
    if False:
        indivs_cons = ctnr.plot_all_data_all_containers(
            my_instance.df_indiv, sep_time=my_instance.sep_time)
        indivs_cons.savefig(path + '/indivs_cons.svg')

    # Analysis period

    # Get dataframe of current part
    working_df_indiv = my_instance.df_indiv.loc[
        my_instance.df_indiv[it.tick_field] <= my_instance.sep_time
    ]
    df_host_evo = pd.DataFrame(columns=my_instance.df_host.columns)
    nb_overloads = 0
    total_method_time = time.time()

    if method == 'init':
        df_host_evo = my_instance.df_host
    if method in ['heur', 'loop', 'loop_v2', 'loop_kmeans']:
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
        cluster_profiles = clt.get_cluster_mean_profile(
            df_indiv_clust)
        cluster_vars = clt.get_cluster_variance(cluster_profiles)

        cluster_var_matrix = clt.get_sum_cluster_variance(
            cluster_profiles, cluster_vars)

        it.results_file.write('\nClustering computing time : %f s\n\n' %
                              (time.time() - clustering_time))

        # Placement
        if config['placement']['enable']:
            logging.info('Performing placement ... \n')
            heur_time = time.time()
            place.allocation_distant_pairwise(
                my_instance, cluster_var_matrix, labels_)
            heur_time = time.time() - heur_time
        else:
            logging.info('We do not perform placement \n')

        if method in ['loop', 'loop_v2', 'loop_kmeans']:
            # loop 'streaming' progress
            it.results_file.write('\n### Loop process ###\n')
            (fig_node, fig_clust, fig_mean_clust,
             main_results, df_host_evo, nb_overloads) = streaming_eval(
                my_instance, df_indiv_clust, labels_,
                config['loop']['mode'],
                config['loop']['tick'],
                config['loop']['constraints_dual'],
                config['loop']['tol_dual_clust'],
                config['loop']['tol_move_clust'],
                config['loop']['tol_dual_place'],
                config['loop']['tol_move_place'],
                config['loop']['tol_step'],
                method,
                df_host_evo)
            fig_node.savefig(output_path + '/node_evo_plot.svg')
            fig_clust.savefig(output_path + '/clust_evo_plot.svg')
            fig_mean_clust.savefig(output_path + '/mean_clust_evo_plot.svg')
    elif method == 'spread':
        place.allocation_spread(my_instance, my_instance.nb_nodes)
    elif method == 'iter-consol':
        place.allocation_spread(my_instance)
    if method in ['heur', 'spread', 'iter-consol']:
        (df_host_evo, nb_overloads) = progress_time_noloop(
            my_instance, my_instance.sep_time, my_instance.df_indiv[it.tick_field].max())
    total_method_time = time.time() - total_method_time

    # Print objectives of evaluation part
    (obj_nodes, obj_delta) = mc.get_obj_value_host(df_host_evo)
    it.results_file.write('Number of nodes : %d, Delta : %f\n' % (
        obj_nodes, obj_delta))
    it.results_file.write('Number of overloads : %d\n' % nb_overloads)
    it.results_file.write('Total execution time : %f s\n\n' % (total_method_time))

    it.clustering_file.write('\nFinal k : %d' % my_instance.nb_clusters)

    # Save evaluation results in files
    it.main_results.append(obj_nodes)
    it.main_results.append(obj_delta)
    node_results = node.get_nodes_load_info(
        df_host_evo, my_instance.df_host_meta)

    # Plot nodes consumption
    # TODO plot from df_host_evo
    # node_evo_fig = plot.plot_containers_groupby_nodes(
    #     my_instance.df_indiv,
    #     my_instance.df_host_meta[it.metrics[0]].max(),
    #     my_instance.sep_time,
    #     title='Initial Node consumption')
    # node_evo_fig.savefig(output_path + '/init_node_plot.svg')

    # Plot clustering & allocation for 1st part
    # plot_before_loop = False
    # if plot_before_loop:
    #     spec_containers = False
    #     if spec_containers:
    #         ctnr.show_specific_containers(working_df_indiv, df_indiv_clust,
    #                                       my_instance, labels_)
    #     show_clustering = True
    #     if show_clustering:
    #         working_df_indiv = my_instance.df_indiv.loc[
    #             my_instance.df_indiv[it.tick_field] <= my_instance.sep_time
    #         ]
    #         clust_node_fig = plot.plot_clustering_containers_by_node(
    #             working_df_indiv, my_instance.dict_id_c,
    #             labels_, filter_big=True)
    #         clust_node_fig.savefig(output_path + '/clust_node_plot.svg')
    #         first_clust_fig = plot.plot_clustering(df_indiv_clust, my_instance.dict_id_c,
    #                                                title='Clustering on first half part')
    #         first_clust_fig.savefig(output_path + '/first_clust_plot.svg')

    # Test allocation use case
    # TODO specific window not taken into account
    # if config['allocation']['enable']:
    #     logging.info('Performing allocation ... \n')
    #     print(alloc.check_constraints(
    #         my_instance, working_df_indiv, config['allocation']))
    # else:
    #     logging.info('We do not perform allocation \n')

    main_time = time.time() - main_time
    it.main_results.append(main_time)
    node_results.to_csv(output_path + '/node_results.csv')
    it.loop_results.to_csv(output_path + '/loop_results.csv', index=False)
    write_main_results()
    it.results_file.write('\nTotal computing time : %f s' % (main_time))
    close_files()


def streaming_eval(my_instance: Instance, df_indiv_clust: pd.DataFrame,
                   labels_: List, mode: str, tick: int,
                   constraints_dual: List,
                   tol_clust: float, tol_move_clust: float,
                   tol_place: float, tol_move_place: float, tol_step: float,
                   method: str, df_host_evo: pd.DataFrame
                   ) -> (plt.Figure, plt.Figure, plt.Figure, List, pd.DataFrame, int):
    """Define the streaming process for evaluation."""
    fig_node, ax_node = plot.init_nodes_plot(
        my_instance.df_indiv, my_instance.dict_id_n, my_instance.sep_time,
        my_instance.df_host_meta[it.metrics[0]].max()
    )
    fig_clust, ax_clust = plot.init_plot_clustering(
        df_indiv_clust, my_instance.dict_id_c)

    cluster_profiles = clt.get_cluster_mean_profile(df_indiv_clust)
    fig_mean_clust, ax_mean_clust = plot.init_plot_cluster_profiles(
        cluster_profiles
    )

    tmin = my_instance.sep_time - tick
    if mode == 'event':
        tmax = my_instance.df_indiv[it.tick_field].max()
    else:
        tmax = my_instance.sep_time
    clustering_dual_values = {}
    placement_dual_values = {}

    total_loop_time = 0.0
    loop_nb = 1
    nb_clust_changes = 0
    nb_place_changes = 0
    total_nb_overload = 0
    end = False

    it.results_file.write('Loop mode : %s\n' % mode)
    logging.info('Beginning the loop process ...\n')
    # TODO improve cplex model builds
    while not end:
        loop_time = time.time()
        logging.info('\n # Enter loop number %d #\n' % loop_nb)
        it.results_file.write('\n # Loop number %d #\n' % loop_nb)
        it.optim_file.write('\n # Enter loop number %d #\n' % loop_nb)
        print('\n # Enter loop number %d #\n' % loop_nb)

        # TODO not fully tested (replace containers)
        if loop_nb > 1:
            temp_time = time.time()
            (temp_df_host, nb_overload) = progress_time_noloop(
                my_instance, 'local', tmin, tmax)
            df_host_evo = df_host_evo.append(
                temp_df_host[~temp_df_host[it.tick_field].isin(
                    df_host_evo[it.tick_field].unique())], ignore_index=True)
            print('Progress no loop time : %f s' % (time.time() - temp_time))
            total_nb_overload += nb_overload

        working_df_indiv = my_instance.df_indiv[
            (my_instance.
                df_indiv[it.tick_field] >= tmin) & (
                my_instance.df_indiv[it.tick_field] <= tmax)]
        (df_clust, my_instance.dict_id_c) = clt.build_matrix_indiv_attr(
            working_df_indiv)
        w = clt.build_similarity_matrix(df_clust)
        df_clust['cluster'] = labels_
        u = clt.build_adjacency_matrix(labels_)
        v = place.build_placement_adj_matrix(
            working_df_indiv, my_instance.dict_id_c)

        if loop_nb == 1:
            logging.info('Evaluation of problems with initial solutions')
            cplex_model = mc.CPXInstance(working_df_indiv,
                                         my_instance.df_host_meta,
                                         my_instance.nb_clusters,
                                         my_instance.dict_id_c,
                                         my_instance.dict_id_n,
                                         w=w, u=u, pb_number=2)
            logging.info('# Clustering evaluation #')
            logging.info('Solving linear relaxation ...')
            cplex_model.solve(cplex_model.relax_mdl)
            logging.info('Clustering problem not evaluated yet\n')
            clustering_dual_values = mc.fill_constraints_dual_values(
                cplex_model.relax_mdl, constraints_dual
            )

            # TODO improve this part (distance...)
            cluster_profiles = clt.get_cluster_mean_profile(
                df_clust)
            cluster_vars = clt.get_cluster_variance(cluster_profiles)

            cluster_var_matrix = clt.get_sum_cluster_variance(
                cluster_profiles, cluster_vars)
            dv = ctnr.build_var_delta_matrix_cluster(
                df_clust, cluster_var_matrix, my_instance.dict_id_c)

            # evaluate placement
            logging.info('# Placement evaluation #')
            cplex_model = mc.CPXInstance(working_df_indiv,
                                         my_instance.df_host_meta,
                                         my_instance.nb_clusters,
                                         my_instance.dict_id_c,
                                         my_instance.dict_id_n,
                                         w=w, u=u, v=v, dv=dv, pb_number=3)
            print('Solving linear relaxation ...')
            cplex_model.solve(cplex_model.relax_mdl)
            logging.info('Placement problem not evaluated yet\n')
            placement_dual_values = mc.fill_constraints_dual_values(
                cplex_model.relax_mdl, constraints_dual
            )
            if mode == 'event':
                (temp_df_host, nb_overload, loop_nb,
                 nb_clust_changes, nb_place_changes) = progress_time_noloop(
                    my_instance, 'loop', tmin, tmax, labels_, loop_nb,
                    constraints_dual, clustering_dual_values, placement_dual_values,
                    tol_clust, tol_move_clust, tol_place, tol_move_place)
                df_host_evo = df_host_evo.append(
                    temp_df_host[~temp_df_host[it.tick_field].isin(
                        df_host_evo[it.tick_field].unique())], ignore_index=True)
                total_nb_overload += nb_overload

        else:  # We have already an evaluated solution

            nb_clust_changes_loop = 0
            nb_place_changes_loop = 0

            (init_loop_obj_nodes, init_loop_obj_delta) = mc.get_obj_value_heuristic(
                working_df_indiv)

            # TODO not very practical
            # plot.plot_clustering_containers_by_node(
            #     working_df_indiv, my_instance.dict_id_c, labels_)

            # eval & change clustering + placement with old method
            # (nb_clust_changes_loop, nb_place_changes_loop,
            #  clustering_dual_values, placement_dual_values) = eval_sols_old(
            #     my_instance, working_df_indiv,
            #     w, u, v, dv,
            #     constraints_dual, clustering_dual_values, placement_dual_values,
            #     tol_clust, tol_move_clust, tol_place, tol_move_place,
            #     df_clust, cluster_profiles, labels_
            # )

            (nb_clust_changes_loop, nb_place_changes_loop,
             clustering_dual_values, placement_dual_values,
             df_clust, cluster_profiles, labels_) = eval_sols(
                my_instance, working_df_indiv,
                w, u, v, dv,
                constraints_dual, clustering_dual_values, placement_dual_values,
                tol_clust, tol_move_clust, tol_place, tol_move_place,
                df_clust, cluster_profiles, labels_
            )

            it.results_file.write('Number of changes in clustering : %d\n' % nb_clust_changes_loop)
            it.results_file.write('Number of changes in placement : %d\n' % nb_place_changes_loop)
            nb_clust_changes += nb_clust_changes_loop
            nb_place_changes += nb_place_changes_loop

            # update clustering & node consumption plot
            plot.update_clustering_plot(
                fig_clust, ax_clust, df_clust, my_instance.dict_id_c)
            plot.update_cluster_profiles(fig_mean_clust, ax_mean_clust, cluster_profiles,
                                         sorted(working_df_indiv[it.tick_field].unique()))
            plot.update_nodes_plot(fig_node, ax_node,
                                   working_df_indiv, my_instance.dict_id_n)

            working_df_indiv = my_instance.df_indiv[
                (my_instance.
                    df_indiv[it.tick_field] >= tmin) & (
                    my_instance.df_indiv[it.tick_field] <= tmax)]
            working_df_host = working_df_indiv.groupby(
                [working_df_indiv[it.tick_field], it.host_field],
                as_index=False).agg(it.dict_agg_metrics)

            if loop_nb > 1:
                df_host_evo = df_host_evo.append(
                    working_df_host[~working_df_host[it.tick_field].isin(
                        df_host_evo[it.tick_field].unique())], ignore_index=True)

            loop_time = (time.time() - loop_time)
            (end_loop_obj_nodes, end_loop_obj_delta) = mc.get_obj_value_host(
                working_df_host)
            it.results_file.write('Loop delta before changes : %f\n' % init_loop_obj_delta)
            it.results_file.write('Loop delta after changes : %f\n' % end_loop_obj_delta)
            it.results_file.write('Loop time : %f s\n' % loop_time)
            total_loop_time += loop_time

            # Save loop indicators in df
            it.loop_results = it.loop_results.append({
                'num_loop': int(loop_nb),
                'init_delta': init_loop_obj_delta,
                'clust_changes': int(nb_clust_changes_loop),
                'place_changes': int(nb_place_changes_loop),
                'end_delta': end_loop_obj_delta,
                'loop_time': loop_time
            }, ignore_index=True)

        print(it.loop_results)
        print(tmin, tmax, tick)
        # input('\nPress any key to progress in time ...\n')
        tmin += tick
        tmax += tick
        if tol_clust < 1.0:
            tol_clust += tol_step
        if tol_place < 1.0:
            tol_place += tol_step

        if tmax >= my_instance.time:
            end = True
        else:
            loop_nb += 1

    working_df_indiv = my_instance.df_indiv[
        (my_instance.
         df_indiv[it.tick_field] >= tmin)]
    if tmin < working_df_indiv[it.tick_field].max():
        # update clustering & node consumption plot
        # TODO not same size issue with clustering
        # print(cluster_profiles)
        # plot.update_clustering_plot(
        #     fig_clust, ax_clust, df_clust, my_instance.dict_id_c)
        # plot.update_cluster_profiles(fig_mean_clust, ax_mean_clust, cluster_profiles,
        #                              sorted(working_df_indiv[it.tick_field].unique()))
        plot.update_nodes_plot(fig_node, ax_node,
                               working_df_indiv, my_instance.dict_id_n)

    df_host_evo = end_loop(working_df_indiv, tmin, nb_clust_changes, nb_place_changes,
                           total_nb_overload, total_loop_time, loop_nb, df_host_evo)

    return (fig_node, fig_clust, fig_mean_clust,
            [nb_clust_changes, nb_place_changes, total_loop_time, total_loop_time / loop_nb],
            df_host_evo, total_nb_overload)


def progress_time_noloop(
        instance: Instance, fixing: str, tmin: int, tmax: int, labels_, loop_nb,
        constraints_dual, clustering_dual_values, placement_dual_values,
        tol_clust, tol_move_clust, tol_place, tol_move_place) -> (pd.DataFrame, int):
    """We progress in time without performing the loop, checking node capacities."""
    df_host_evo = pd.DataFrame(columns=instance.df_host.columns)
    nb_overload = 0
    nb_clust_changes = 0
    nb_place_changes = 0

    for tick in range(tmin, tmax + 1):
        df_host_tick = instance.df_indiv.loc[
            instance.df_indiv[it.tick_field] == tick
        ].groupby(
            [instance.df_indiv[it.tick_field], it.host_field],
            as_index=False).agg(it.dict_agg_metrics)

        host_overload = node.check_capacities(df_host_tick, instance.df_host_meta)
        df_host_tick[it.tick_field] = tick
        df_host_evo = df_host_evo.append(
            df_host_tick, ignore_index=True)
        if len(host_overload) > 0:
            print('Overload : We must move containers')
            nb_overload += len(host_overload)
            if fixing == 'local':
                place.free_full_nodes(instance, host_overload, tick)
            elif fixing == 'loop':
                working_df_indiv = instance.df_indiv[
                    (instance.
                     df_indiv[it.tick_field] >= tmin) & (
                        instance.df_indiv[it.tick_field] <= tmax)]
                (df_clust, instance.dict_id_c) = clt.build_matrix_indiv_attr(
                    working_df_indiv)
                w = clt.build_similarity_matrix(df_clust)
                df_clust['cluster'] = labels_
                u = clt.build_adjacency_matrix(labels_)
                v = place.build_placement_adj_matrix(
                    working_df_indiv, instance.dict_id_c)
                cluster_profiles = clt.get_cluster_mean_profile(
                    df_clust)
                cluster_vars = clt.get_cluster_variance(cluster_profiles)

                cluster_var_matrix = clt.get_sum_cluster_variance(
                    cluster_profiles, cluster_vars)
                dv = ctnr.build_var_delta_matrix_cluster(
                    df_clust, cluster_var_matrix, instance.dict_id_c)
                (nb_clust_changes_loop, nb_place_changes_loop,
                 clustering_dual_values, placement_dual_values,
                 df_clust, cluster_profiles, labels_) = eval_sols(
                    instance, working_df_indiv,
                    w, u, v, dv,
                    constraints_dual, clustering_dual_values, placement_dual_values,
                    tol_clust, tol_move_clust, tol_place, tol_move_place,
                    df_clust, cluster_profiles, labels_
                )
                if nb_place_changes_loop < 1:
                    place.free_full_nodes(instance, host_overload, tick)
                loop_nb += 1
                nb_clust_changes += nb_clust_changes_loop
                nb_place_changes += nb_place_changes_loop
    return (df_host_evo, nb_overload, loop_nb, nb_clust_changes, nb_place_changes)


def eval_sols(
        my_instance: Instance, working_df_indiv: pd.DataFrame,
        w: np.array, u, v, dv,
        constraints_dual, clustering_dual_values, placement_dual_values,
        tol_clust, tol_move_clust, tol_place, tol_move_place,
        df_clust, cluster_profiles, labels_,
        method: str = 'loop'
):
    """Evaluate clustering and placement solutions with method technique."""
    if method == 'loop_kmeans':
        return eval_sol_scratch(
            my_instance, working_df_indiv,
            w, u, v, dv,
            constraints_dual, clustering_dual_values, placement_dual_values,
            tol_clust, tol_move_clust, tol_place, tol_move_place,
            df_clust, cluster_profiles, labels_)
    elif method == 'loop_v2':
        return eval_sols_new(
            my_instance, working_df_indiv,
            w, u, v, dv,
            constraints_dual, clustering_dual_values, placement_dual_values,
            tol_clust, tol_move_clust, tol_place, tol_move_place,
            df_clust, cluster_profiles, labels_)
    else:
        return eval_sols_old(
            my_instance, working_df_indiv,
            w, u, v, dv,
            constraints_dual, clustering_dual_values, placement_dual_values,
            tol_clust, tol_move_clust, tol_place, tol_move_place,
            df_clust, cluster_profiles, labels_)


def eval_sols_old(
        my_instance: Instance, working_df_indiv: pd.DataFrame,
        w: np.array, u, v, dv,
        constraints_dual, clustering_dual_values, placement_dual_values,
        tol_clust, tol_move_clust, tol_place, tol_move_place,
        df_clust, cluster_profiles, labels_):
    """Evaluate and update solutions using old method."""
    # evaluate clustering
    (dv, nb_clust_changes_loop,
        clustering_dual_values) = eval_clustering(
        my_instance, working_df_indiv,
        w, u, clustering_dual_values, constraints_dual,
        tol_clust, tol_move_clust,
        df_clust, cluster_profiles, labels_)

    # evaluate placement
    (nb_place_changes_loop,
        placement_dual_values) = eval_placement(
        my_instance, working_df_indiv,
        w, u, v, dv,
        placement_dual_values, constraints_dual,
        tol_place, tol_move_place, nb_clust_changes_loop
    )

    return (
        nb_clust_changes_loop, nb_place_changes_loop,
        clustering_dual_values, placement_dual_values,
        df_clust, cluster_profiles, labels_
    )


def eval_sol_scratch(
    my_instance: Instance, working_df_indiv: pd.DataFrame,
        w: np.array, u, v, dv,
        constraints_dual, clustering_dual_values, placement_dual_values,
        tol_clust, tol_move_clust, tol_place, tol_move_place,
        df_clust, cluster_profiles, labels_
):
    """Evaluate and update placement solution after new kmeans clustering."""
    (ics, icd) = clt.eval_clustering(df_clust, w, my_instance.dict_id_c)
    it.clustering_file.write('ICS and ICD before any change : %f, %f\n' % (ics, icd))
    print('ICS and ICD before any change : %f, %f\n' % (ics, icd))

    (df_clust_scratch, my_instance.dict_id_c) = clt.build_matrix_indiv_attr(
        working_df_indiv)
    labels_scratch = clt.perform_clustering(
        df_clust_scratch, 'kmeans', my_instance.nb_clusters)
    df_clust_scratch['cluster'] = labels_scratch
    (ics, icd) = clt.eval_clustering(df_clust_scratch, w, my_instance.dict_id_c)
    it.clustering_file.write(
        'ICS and ICD after new k-means : %f, %f\n' % (ics, icd))
    # Compute new clusters profiles
    cluster_profiles = clt.get_cluster_mean_profile(
        df_clust)
    cluster_vars = clt.get_cluster_variance(cluster_profiles)
    cluster_var_matrix = clt.get_sum_cluster_variance(
        cluster_profiles, cluster_vars)
    dv = ctnr.build_var_delta_matrix_cluster(
        df_clust, cluster_var_matrix, my_instance.dict_id_c)

    nb_clust_changes_loop = 1
    # evaluate placement
    (nb_place_changes_loop,
        placement_dual_values) = eval_placement(
        my_instance, working_df_indiv,
        w, u, v, dv,
        placement_dual_values, constraints_dual,
        tol_place, tol_move_place, nb_clust_changes_loop
    )

    return (
        nb_clust_changes_loop, nb_place_changes_loop,
        clustering_dual_values, placement_dual_values,
        df_clust, cluster_profiles, labels_
    )


def eval_sols_new(
        my_instance: Instance, working_df_indiv: pd.DataFrame,
        w: np.array, u, v, dv,
        constraints_dual, clustering_dual_values, placement_dual_values,
        tol_clust, tol_move_clust, tol_place, tol_move_place,
        df_clust, cluster_profiles, labels_):
    """Evaluate and update solutions using new method."""
    # evaluate clustering
    (nb_clust_changes_loop,
        clustering_dual_values, constraints_rm) = eval_clustering_v2(
        my_instance, working_df_indiv,
        w, u, clustering_dual_values, constraints_dual,
        tol_clust, tol_move_clust,
        df_clust, cluster_profiles, labels_)
    # evaluate placement
    (nb_place_changes_loop,
        placement_dual_values, confirm_change_clust) = eval_placement_v2(
        my_instance, working_df_indiv,
        w, u, v, dv,
        placement_dual_values, constraints_dual, constraints_rm,
        tol_place, tol_move_place, nb_clust_changes_loop
    )

    if confirm_change_clust:
        moving_containers = mc.get_mvg_conts_from_constraints(
            constraints_rm, my_instance.dict_id_c,
            df_clust, cluster_profiles)
        if len(moving_containers) > 0:
            logging.info('Changing clustering ...')
            time_change_clust = time.time()
            (df_clust, labels_, nb_clust_changes_loop) = clt.change_clustering(
                moving_containers, df_clust, labels_,
                my_instance.dict_id_c)
            u = clt.build_adjacency_matrix(labels_)
            print('Time changing clustering : %f s' % (time.time() - time_change_clust))
            (ics, icd) = clt.eval_clustering(df_clust, w, my_instance.dict_id_c)
            it.clustering_file.write(
                'ICS and ICD after loop change : %f, %f\n' % (ics, icd))
            print(
                'ICS and ICD after loop change : %f, %f\n' % (ics, icd))
        # it.clustering_file.write(
        #     'Loop clustering time : %f s\n' % (time.time() - time_loop_clust))
        cplex_model = mc.CPXInstance(working_df_indiv,
                                     my_instance.df_host_meta,
                                     my_instance.nb_clusters,
                                     my_instance.dict_id_c,
                                     my_instance.dict_id_n,
                                     w=w, u=u, pb_number=2)
        logging.info('Solving linear relaxation after changes ...')
        cplex_model.solve(cplex_model.relax_mdl)
        print('After changes clustering lp solution : ', cplex_model.relax_mdl.objective_value)
        clustering_dual_values = mc.fill_constraints_dual_values(
            cplex_model.relax_mdl, constraints_dual
        )
    else:
        logging.info('Clustering seems still right ...')
        it.results_file.write('Clustering seems still right ...')

    cluster_profiles = clt.get_cluster_mean_profile(
        df_clust)
    cluster_vars = clt.get_cluster_variance(cluster_profiles)
    cluster_var_matrix = clt.get_sum_cluster_variance(
        cluster_profiles, cluster_vars)
    dv = ctnr.build_var_delta_matrix_cluster(
        df_clust, cluster_var_matrix, my_instance.dict_id_c)

    return (
        nb_clust_changes_loop, nb_place_changes_loop,
        clustering_dual_values, placement_dual_values,
        df_clust, cluster_profiles, labels_
    )


def eval_clustering(my_instance: Instance, working_df_indiv: pd.DataFrame,
                    w: np.array, u: np.array,
                    clustering_dual_values: Dict, constraints_dual: Dict,
                    tol_clust: float, tol_move_clust: float,
                    df_clust: pd.DataFrame, cluster_profiles: np.array, labels_: List) -> np.array:
    """Evaluate current clustering solution and update it if needed."""
    nb_clust_changes_loop = 0

    (ics, icd) = clt.eval_clustering(df_clust, w, my_instance.dict_id_c)
    it.clustering_file.write('ICS and ICD before any change : %f, %f\n' % (ics, icd))
    print('ICS and ICD before any change : %f, %f\n' % (ics, icd))

    time_loop_clust = time.time()
    # TODO update model from existing one, not creating new one each time
    cplex_model = mc.CPXInstance(working_df_indiv,
                                 my_instance.df_host_meta,
                                 my_instance.nb_clusters,
                                 my_instance.dict_id_c,
                                 my_instance.dict_id_n,
                                 w=w, u=u, pb_number=2)
    logging.info('# Clustering evaluation #')
    logging.info('Solving linear relaxation ...')
    cplex_model.solve(cplex_model.relax_mdl)
    print('Init clustering lp solution : ', cplex_model.relax_mdl.objective_value)

    logging.info('Checking for changes in clustering dual values ...')
    time_get_clust_move = time.time()
    moving_containers = mc.get_moving_containers_clust(
        cplex_model.relax_mdl, clustering_dual_values,
        tol_clust, tol_move_clust,
        my_instance.nb_containers, my_instance.dict_id_c,
        df_clust, cluster_profiles)
    print('Time get changing clustering : %f s' % (time.time() - time_get_clust_move))
    if len(moving_containers) > 0:
        logging.info('Changing clustering ...')
        time_change_clust = time.time()
        (df_clust, labels_, nb_clust_changes_loop) = clt.change_clustering(
            moving_containers, df_clust, labels_,
            my_instance.dict_id_c)
        u = clt.build_adjacency_matrix(labels_)
        print('Time changing clustering : %f s' % (time.time() - time_change_clust))
        (ics, icd) = clt.eval_clustering(df_clust, w, my_instance.dict_id_c)
        it.clustering_file.write(
            'ICS and ICD after loop change : %f, %f\n' % (ics, icd))
        print(
            'ICS and ICD after loop change : %f, %f\n' % (ics, icd))
        it.clustering_file.write(
            'Loop clustering time : %f s\n' % (time.time() - time_loop_clust))

        # Perform k-means from scratch (compare with loop changing)
        time_kmeans = time.time()
        (df_clust_scratch, my_instance.dict_id_c) = clt.build_matrix_indiv_attr(
            working_df_indiv)
        labels_scratch = clt.perform_clustering(
            df_clust_scratch, 'kmeans', my_instance.nb_clusters)
        df_clust_scratch['cluster'] = labels_scratch
        (ics, icd) = clt.eval_clustering(df_clust_scratch, w, my_instance.dict_id_c)
        it.clustering_file.write(
            'ICS and ICD after new k-means : %f, %f\n' % (ics, icd))
        it.clustering_file.write(
            'Kmeans clustering time : %f s\n\n' % (time.time() - time_kmeans))

        cplex_model = mc.CPXInstance(working_df_indiv,
                                     my_instance.df_host_meta,
                                     my_instance.nb_clusters,
                                     my_instance.dict_id_c,
                                     my_instance.dict_id_n,
                                     w=w, u=u, pb_number=2)
        logging.info('Solving linear relaxation after changes ...')
        cplex_model.solve(cplex_model.relax_mdl)
        clustering_dual_values = mc.fill_constraints_dual_values(
            cplex_model.relax_mdl, constraints_dual
        )
        print('After changes clustering lp solution : ', cplex_model.relax_mdl.objective_value)
    else:
        logging.info('Clustering seems still right ...')
        it.results_file.write('Clustering seems still right ...')

    # TODO improve this part (use cluster profiles)
    # temp_time = time.time()
    # dv = ctnr.build_var_delta_matrix(
    #     working_df_indiv, my_instance.dict_id_c)
    # print('Building var_delta matrices time : %f s' % (time.time() - temp_time))

    # TODO improve this part (distance...)
    # Compute new clusters profiles
    cluster_profiles = clt.get_cluster_mean_profile(
        df_clust)
    cluster_vars = clt.get_cluster_variance(cluster_profiles)
    cluster_var_matrix = clt.get_sum_cluster_variance(
        cluster_profiles, cluster_vars)
    dv = ctnr.build_var_delta_matrix_cluster(
        df_clust, cluster_var_matrix, my_instance.dict_id_c)

    return (dv, nb_clust_changes_loop, clustering_dual_values)


def eval_placement(my_instance: Instance, working_df_indiv: pd.DataFrame,
                   w: np.array, u: np.array, v: np.array, dv: np.array,
                   placement_dual_values: Dict, constraints_dual: Dict,
                   tol_place: float, tol_move_place: float,
                   nb_clust_changes_loop: int):
    """Evaluate current clustering solution and update it if needed."""
    logging.info('# Placement evaluation #')
    cplex_model = mc.CPXInstance(working_df_indiv,
                                 my_instance.df_host_meta,
                                 my_instance.nb_clusters,
                                 my_instance.dict_id_c,
                                 my_instance.dict_id_n,
                                 w=w, u=u, v=v, dv=dv, pb_number=3)
    print('Solving linear relaxation ...')
    it.optim_file.write('solve without any change\n')
    cplex_model.solve(cplex_model.relax_mdl)
    print('Init placement lp solution : ', cplex_model.relax_mdl.objective_value)
    moving_containers = []
    nb_place_changes_loop = 0

    if nb_clust_changes_loop > 0:
        logging.info('Checking for changes in placement dual values ...')
        time_get_move = time.time()
        moving_containers = mc.get_moving_containers(
            cplex_model.relax_mdl, placement_dual_values,
            tol_place, tol_move_place, my_instance.nb_containers,
            working_df_indiv, my_instance.dict_id_c)
        print('Time get moving containers : %f s' % (time.time() - time_get_move))
        if len(moving_containers) > 0:
            nb_place_changes_loop = len(moving_containers)
            time_move_place = time.time()
            place.move_list_containers(moving_containers, my_instance,
                                       working_df_indiv[it.tick_field].min(),
                                       working_df_indiv[it.tick_field].max())
            print('Time moving containers : %f s' % (time.time() - time_move_place))
            v = place.build_placement_adj_matrix(
                working_df_indiv, my_instance.dict_id_c)
            cplex_model = mc.CPXInstance(working_df_indiv,
                                         my_instance.df_host_meta,
                                         my_instance.nb_clusters,
                                         my_instance.dict_id_c,
                                         my_instance.dict_id_n,
                                         w=w, u=u, v=v, dv=dv, pb_number=3)
            print('Solving linear relaxation after changes ...')
            cplex_model.solve(cplex_model.relax_mdl)
            print('After changes placement lp solution : ', cplex_model.relax_mdl.objective_value)
            placement_dual_values = mc.fill_constraints_dual_values(
                cplex_model.relax_mdl, constraints_dual
            )
        else:
            logging.info('No container to move : we do nothing ...\n')

    return (nb_place_changes_loop, placement_dual_values)


def eval_clustering_v2(my_instance: Instance, working_df_indiv: pd.DataFrame,
                       w: np.array, u: np.array,
                       clustering_dual_values: Dict, constraints_dual: Dict,
                       tol_clust: float, tol_move_clust: float,
                       df_clust: pd.DataFrame, cluster_profiles: np.array, labels_: List
                       ):
    """Evaluate current clustering solution and update it if needed."""
    nb_clust_changes_loop = 0

    (ics, icd) = clt.eval_clustering(df_clust, w, my_instance.dict_id_c)
    it.clustering_file.write('ICS and ICD before any change : %f, %f\n' % (ics, icd))
    print('ICS and ICD before any change : %f, %f\n' % (ics, icd))

    # time_loop_clust = time.time()
    # TODO update model from existing one, not creating new one each time
    cplex_model = mc.CPXInstance(working_df_indiv,
                                 my_instance.df_host_meta,
                                 my_instance.nb_clusters,
                                 my_instance.dict_id_c,
                                 my_instance.dict_id_n,
                                 w=w, u=u, pb_number=2)
    logging.info('# Clustering evaluation #')
    logging.info('Solving linear relaxation ...')
    cplex_model.solve(cplex_model.relax_mdl)
    init_obj = cplex_model.relax_mdl.objective_value
    print('Init clustering lp solution : ', init_obj)

    logging.info('Checking for changes in clustering dual values ...')
    # time_get_clust_move = time.time()
    constraints_rm = mc.get_conflict_graph_clust(
        cplex_model, clustering_dual_values,
        tol_clust, tol_move_clust,
        my_instance.nb_containers, my_instance.dict_id_c,
        df_clust, cluster_profiles
    )
    # moving_containers = mc.get_moving_containers_clust(
    #     cplex_model.relax_mdl, clustering_dual_values,
    #     tol_clust, tol_move_clust,
    #     my_instance.nb_containers, my_instance.dict_id_c,
    #     df_clust, cluster_profiles)
    # print('Time get changing clustering : %f s' % (time.time() - time_get_clust_move))
    # if len(moving_containers) > 0:
    #     logging.info('Changing clustering ...')
    #     time_change_clust = time.time()
    #     (df_clust, labels_, nb_clust_changes_loop) = clt.change_clustering(
    #         moving_containers, df_clust, labels_,
    #         my_instance.dict_id_c)
    #     if nb_clust_changes_loop < 1:
    #         print('mvg_containers non empty but clustering not changed : k++')
    #         new_k = my_instance.nb_clusters + 1
    #         (df_clust_scratch, my_instance.dict_id_c) = clt.build_matrix_indiv_attr(
    #             working_df_indiv)
    #         labels_scratch = clt.perform_clustering(
    #             df_clust_scratch, 'kmeans', new_k)
    #         df_clust_scratch['cluster'] = labels_scratch
    #         u_scratch = clt.build_adjacency_matrix(labels_scratch)
    #         cplex_model = mc.CPXInstance(working_df_indiv,
    #                                      my_instance.df_host_meta,
    #                                      new_k,
    #                                      my_instance.dict_id_c,
    #                                      my_instance.dict_id_n,
    #                                      w=w, u=u_scratch, pb_number=2)
    #         logging.info('# Clustering evaluation #')
    #         logging.info('Solving linear relaxation ...')
    #         cplex_model.solve(cplex_model.relax_mdl)
    #         print('Init clustering lp solution : ', cplex_model.relax_mdl.objective_value)
    #         print('Improvement : ', (init_obj - cplex_model.relax_mdl.objective_value) / init_obj)
    #         if (
    #             (init_obj - cplex_model.relax_mdl.objective_value) / init_obj
    #         ) > tol_clust:
    #             print('Improving better than tol : we keep k=', new_k)
    #             it.results_file.write('\nWe incremented k : %d\n' % new_k)
    #             it.clustering_file.write('\nWe incremented k : %d\n' % new_k)
    #             my_instance.nb_clusters = new_k
    #             df_clust = df_clust_scratch
    #             labels_ = labels_scratch
    #             u = u_scratch
    #             nb_clust_changes_loop += 1

    #     else:
    #         u = clt.build_adjacency_matrix(labels_)
    #         cplex_model = mc.CPXInstance(working_df_indiv,
    #                                      my_instance.df_host_meta,
    #                                      my_instance.nb_clusters,
    #                                      my_instance.dict_id_c,
    #                                      my_instance.dict_id_n,
    #                                      w=w, u=u, pb_number=2)
    #         logging.info('Solving linear relaxation after changes ...')
    #         cplex_model.solve(cplex_model.relax_mdl)
    #         print('Time changing clustering : %f s' % (time.time() - time_change_clust))
    #         (ics, icd) = clt.eval_clustering(df_clust, w, my_instance.dict_id_c)
    #         it.clustering_file.write(
    #             'ICS and ICD after loop change : %f, %f\n' % (ics, icd))
    #         it.clustering_file.write(
    #             'Loop clustering time : %f s\n' % (time.time() - time_loop_clust))
    #     clustering_dual_values = mc.fill_constraints_dual_values(
    #         cplex_model.relax_mdl, constraints_dual
    #     )

    # else:
    #     logging.info('Clustering seems still right ...')
    #     it.results_file.write('Clustering seems still right ...')
    # Perform k-means from scratch (compare with loop changing)
    # time_kmeans = time.time()
    # (df_clust_scratch, my_instance.dict_id_c) = clt.build_matrix_indiv_attr(
    #     working_df_indiv)
    # labels_scratch = clt.perform_clustering(
    #     df_clust_scratch, 'kmeans', my_instance.nb_clusters)
    # df_clust_scratch['cluster'] = labels_scratch
    # (ics, icd) = clt.eval_clustering(df_clust_scratch, w, my_instance.dict_id_c)
    # it.clustering_file.write(
    #     'ICS and ICD after new k-means : %f, %f\n' % (ics, icd))
    # it.clustering_file.write(
    #     'Kmeans clustering time : %f s\n\n' % (time.time() - time_kmeans))

    #     cplex_model = mc.CPXInstance(working_df_indiv,
    #                                  my_instance.df_host_meta,
    #                                  my_instance.nb_clusters,
    #                                  my_instance.dict_id_c,
    #                                  my_instance.dict_id_n,
    #                                  w=w, u=u, pb_number=2)
    #     logging.info('Solving linear relaxation after changes ...')
    #     cplex_model.solve(cplex_model.relax_mdl)
    #     clustering_dual_values = mc.fill_constraints_dual_values(
    #         cplex_model.relax_mdl, constraints_dual
    #     )

    # TODO improve this part (use cluster profiles)
    # temp_time = time.time()
    # dv = ctnr.build_var_delta_matrix(
    #     working_df_indiv, my_instance.dict_id_c)
    # print('Building var_delta matrices time : %f s' % (time.time() - temp_time))

    # TODO improve this part (distance...)
    # Compute new clusters profiles
    # cluster_profiles = clt.get_cluster_mean_profile(
    #     df_clust)
    # cluster_vars = clt.get_cluster_variance(cluster_profiles)
    # cluster_var_matrix = clt.get_sum_cluster_variance(
    #     cluster_profiles, cluster_vars)
    # dv = ctnr.build_var_delta_matrix_cluster(
    #     df_clust, cluster_var_matrix, my_instance.dict_id_c)
    # cv = ctnr.build_vars_matrix_indivs(
    #     df_clust, cluster_vars, my_instance.dict_id_c)

    # return (cv, dv, nb_clust_changes_loop, clustering_dual_values, constraints_rm,
    #         df_clust, cluster_profiles, labels_)
    return (nb_clust_changes_loop, clustering_dual_values, constraints_rm)


def eval_placement_v2(my_instance: Instance, working_df_indiv: pd.DataFrame,
                      w: np.array, u: np.array, v: np.array, dv: np.array,
                      placement_dual_values: Dict, constraints_dual: Dict, constraints_rm,
                      tol_place: float, tol_move_place: float,
                      nb_clust_changes_loop: int):
    """Evaluate current clustering solution and update it if needed."""
    logging.info('# Placement evaluation #')
    confirm_change_clust = False
    cplex_model = mc.CPXInstance(working_df_indiv,
                                 my_instance.df_host_meta,
                                 my_instance.nb_clusters,
                                 my_instance.dict_id_c,
                                 my_instance.dict_id_n,
                                 w=w, u=u, v=v, dv=dv, pb_number=3)
    print('Solving linear relaxation ...')
    it.optim_file.write('solve without any change\n')
    cplex_model.solve(cplex_model.relax_mdl)
    init_obj_place = cplex_model.relax_mdl.objective_value
    print('Init placement lp obj : ', cplex_model.relax_mdl.objective_value)
    moving_containers = []
    nb_place_changes_loop = 0

    # New method (in test) : only constraints to remove are passed
    if len(constraints_rm) > 0:
        u = mc.set_u_constraints_rm(constraints_rm, u)
        cplex_model = mc.CPXInstance(working_df_indiv,
                                     my_instance.df_host_meta,
                                     my_instance.nb_clusters,
                                     my_instance.dict_id_c,
                                     my_instance.dict_id_n,
                                     w=w, u=u, v=v, dv=dv, pb_number=3)
        print('Solving linear relaxation ...')
        it.optim_file.write('new solve after changing u\n')
        cplex_model.solve(cplex_model.relax_mdl)
        adj_u_obj_place = cplex_model.relax_mdl.objective_value
        print('After changing u, placement lp obj : ', adj_u_obj_place)

        if adj_u_obj_place < init_obj_place:
            print('lp relax better with clust change')
            confirm_change_clust = True

            logging.info('Checking for changes in placement dual values ...')
            time_get_move = time.time()
            moving_containers = mc.get_moving_containers(
                cplex_model.relax_mdl, placement_dual_values,
                tol_place, tol_move_place, my_instance.nb_containers,
                working_df_indiv, my_instance.dict_id_c)
            print('Time get moving containers : %f s' % (time.time() - time_get_move))
        else:
            print('we dont improve lp relax with clust')

    if len(moving_containers) > 0:
        nb_place_changes_loop = len(moving_containers)
        time_move_place = time.time()
        place.move_list_containers(moving_containers, my_instance,
                                   working_df_indiv[it.tick_field].min(),
                                   working_df_indiv[it.tick_field].max())
        print('Time moving containers : %f s' % (time.time() - time_move_place))
        v = place.build_placement_adj_matrix(
            working_df_indiv, my_instance.dict_id_c)
        cplex_model = mc.CPXInstance(working_df_indiv,
                                     my_instance.df_host_meta,
                                     my_instance.nb_clusters,
                                     my_instance.dict_id_c,
                                     my_instance.dict_id_n,
                                     w=w, u=u, v=v, dv=dv, pb_number=3)
        print('Solving linear relaxation after changes ...')
        cplex_model.solve(cplex_model.relax_mdl)
        print('After changes placement lp solution : ', cplex_model.relax_mdl.objective_value)
        placement_dual_values = mc.fill_constraints_dual_values(
            cplex_model.relax_mdl, constraints_dual
        )
    else:
        logging.info('No container to move : we do nothing ...\n')

    return (nb_place_changes_loop, placement_dual_values, confirm_change_clust)


def end_loop(working_df_indiv: pd.DataFrame, tmin: int,
             nb_clust_changes: int, nb_place_changes: int, nb_overload: int,
             total_loop_time: float, loop_nb: int, df_host_evo: pd.DataFrame):
    """Perform all stuffs after last loop."""
    working_df_host = working_df_indiv.groupby(
        [working_df_indiv[it.tick_field], it.host_field],
        as_index=False).agg(it.dict_agg_metrics)

    (end_loop_obj_nodes, end_loop_obj_delta) = mc.get_obj_value_host(
        working_df_host)
    it.results_file.write('Final loop delta : %f\n' % end_loop_obj_delta)

    it.results_file.write('\n### Results of loops ###\n')
    it.results_file.write('Total number of changes in clustering : %d\n' % nb_clust_changes)
    it.results_file.write('Total number of changes in placement : %d\n' % nb_place_changes)
    it.results_file.write('Total number of overload : %d\n' % nb_overload)
    it.results_file.write('Average loop time : %f s\n' % (total_loop_time / loop_nb))

    if loop_nb <= 1:
        df_host_evo = working_df_indiv.groupby(
            [working_df_indiv[it.tick_field], it.host_field],
            as_index=False).agg(it.dict_agg_metrics)
    else:
        df_host_evo = df_host_evo.append(
            working_df_host[~working_df_host[it.tick_field].isin(
                df_host_evo[it.tick_field].unique())], ignore_index=True)
    return df_host_evo


def write_main_results():
    """Write the main results in the .csv file."""
    i = 1
    for e in it.main_results:
        if i < len(it.main_results):
            it.main_results_file.write('%f,' % e)
            i += 1
        else:
            it.main_results_file.write('%f' % e)


def close_files():
    """Write the final files and close all open files."""
    it.results_file.close()
    it.main_results_file.close()


if __name__ == '__main__':
    main()
