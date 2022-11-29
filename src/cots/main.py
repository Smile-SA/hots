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

import logging
import math
import time
from typing import Dict, List, Tuple

import click

from clusopt_core.cluster import Streamkm

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

# Personnal imports
# from . import allocation as alloc
from . import clustering as clt
from . import container as ctnr
from . import init as it
from . import model as mdl
from . import node
from . import placement as place
from . import plot
from .instance import Instance


@click.command()
@click.argument('path', required=True, type=click.Path(exists=True))
@click.option('-k', required=False, type=int, help='Number of clusters')
@click.option('-t', '--tau', required=False, type=int, help='Time window size')
@click.option('-m', '--method', required=False, type=str, default='loop', help='Method used')
@click.option('-c', '--cluster_method', required=False, type=str, default='loop-cluster',
              help='Method used for updating clustering')
@click.option('-p', '--param', required=False, type=str, help='Use a specific parameter file')
@click.option('-o', '--output', required=False, type=str,
              help='Use a specific directory for output')
@click.option('-ec', '--tolclust', required=False, type=str,
              help='Use specific value for epsilonC')
@click.option('-ea', '--tolplace', required=False, type=str,
              help='Use specific value for epsilonA')
def main(path, k, tau, method, cluster_method, param, output, tolclust, tolplace):
    """Use method to propose a placement solution for micro-services adjusted in time."""
    # Initialization part
    main_time = time.time()

    if not path[-1] == '/':
        path += '/'

    start = time.time()
    (config, output_path, my_instance) = preprocess(
        path, k, tau, method, cluster_method, param, output, tolclust, tolplace
    )
    add_time(-1, 'preprocess', (time.time() - start))

    # Plot initial data
    if False:
        indivs_cons = ctnr.plot_all_data_all_containers(
            my_instance.df_indiv, sep_time=my_instance.sep_time)
        indivs_cons.savefig(path + '/indivs_cons.svg')
        node_evo_fig = plot.plot_containers_groupby_nodes(
            my_instance.df_indiv,
            my_instance.df_host_meta[it.metrics[0]].max(),
            my_instance.sep_time,
            title='Initial Node consumption')
        node_evo_fig.savefig(path + '/init_node_plot.svg')
    total_method_time = time.time()

    # Analysis period
    start = time.time()
    (my_instance, df_host_evo,
     df_indiv_clust, labels_) = analysis_period(
        my_instance, config, method
    )
    add_time(-1, 'total_t_obs', (time.time() - start))

    # Run period
    start = time.time()
    (df_host_evo, nb_overloads) = run_period(
        my_instance, df_host_evo,
        df_indiv_clust, labels_,
        config, output_path, method, cluster_method
    )
    add_time(-1, 'total_t_run', (time.time() - start))
    total_method_time = time.time() - total_method_time

    # Print objectives of evaluation part
    (obj_nodes, obj_delta) = mdl.get_obj_value_host(df_host_evo)
    it.results_file.write('Number of nodes : %d, Ampli max : %f\n' % (
        obj_nodes, obj_delta))
    it.results_file.write('Number of overloads : %d\n' % nb_overloads)
    it.results_file.write('Total execution time : %f s\n\n' % (total_method_time))

    it.clustering_file.write('\nFinal k : %d' % my_instance.nb_clusters)

    # Save evaluation results in files
    node_results = node.get_nodes_load_info(
        df_host_evo, my_instance.df_host_meta)
    it.global_results = pd.DataFrame([{
        'c1': obj_nodes,
        'c2': obj_delta,
        'c3': node_results['load_var'].mean(),
        'c4': nb_overloads
    }])

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
    add_time(-1, 'total_time', main_time)
    node.plot_data_all_nodes(
        df_host_evo, it.metrics[0],
        my_instance.df_host_meta[it.metrics[0]].max(),
        my_instance.sep_time).savefig(
        output_path + '/node_usage_evo.svg')
    df_host_evo.to_csv(output_path + '/node_usage_evo.csv', index=False)
    it.global_results.to_csv(output_path + '/global_results.csv', index=False)
    node_results.to_csv(output_path + '/node_results.csv')
    it.loop_results.to_csv(output_path + '/loop_results.csv', index=False)
    it.times_df.to_csv(output_path + '/times.csv', index=False)
    it.results_file.write('\nTotal computing time : %f s' % (main_time))
    close_files()


def preprocess(
        path: str, k: int, tau: int, method: str, cluster_method: str,
        param: str, output: str, tolclust: float, tolplace: float
) -> Tuple[Dict, str, Instance]:
    """Load configuration, data and initialize needed objects."""
    # TODO what if we want tick < tau ?
    print('Preprocessing ...')
    (config, output_path) = it.read_params(path, k, tau, method, cluster_method, param, output)
    config = spec_params(config, [tolclust, tolplace])
    logging.basicConfig(filename=output_path + '/logs.log', filemode='w',
                        format='%(message)s', level=logging.INFO)
    plt.style.use('bmh')

    # Init containers & nodes data, then Instance
    logging.info('Loading data and creating Instance (Instance information are in results file)\n')
    instance = Instance(path, config)
    it.results_file.write('Method used : %s\n' % method)
    instance.print_times(config['loop']['tick'])

    return (config, output_path, instance)


# TODO use Dict instead of List for genericity ?
def spec_params(config: Dict, list_params: List) -> Dict:
    """Define specific parameters."""
    if list_params[0] is not None:
        config['loop']['tol_dual_clust'] = float(list_params[0])
    if list_params[1] is not None:
        config['loop']['tol_dual_place'] = float(list_params[1])
    return config


def analysis_period(
    my_instance: Instance, config: Dict, method: str,
) -> Tuple[Instance, pd.DataFrame, pd.DataFrame, List]:
    """Perform all needed process during analysis period (T_init)."""
    df_host_evo = pd.DataFrame(columns=my_instance.df_host.columns)
    df_indiv_clust = pd.DataFrame()
    labels_ = []

    if method == 'init':
        return (my_instance, my_instance.df_host,
                df_indiv_clust, labels_)
    elif method in ['heur', 'loop']:

        # Compute starting point
        n_iter = math.floor((
            my_instance.sep_time + 1 - my_instance.df_host[it.tick_field].min()
        ) / my_instance.window_duration)
        start_point = (
            my_instance.sep_time - n_iter * my_instance.window_duration
        ) + 1
        end_point = (
            start_point + my_instance.window_duration - 1
        )
        working_df_indiv = my_instance.df_indiv[
            (my_instance.
                df_indiv[it.tick_field] >= start_point) & (
                my_instance.df_indiv[it.tick_field] <= end_point)
        ]

        # First clustering part
        logging.info('Starting first clustering ...')
        print('Starting first clustering ...')
        start = time.time()
        (df_indiv_clust, my_instance.dict_id_c) = clt.build_matrix_indiv_attr(
            working_df_indiv)

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
                              (time.time() - start))
        print('\nClustering computing time : %f s\n\n' %
              (time.time() - start))
        add_time(-1, 'clustering_0', (time.time() - start))
        n_iter = n_iter - 1

        # Loop for incrementing clustering (during analysis)
        # TODO finish
        # while n_iter > 0:
        #     start_point = end_point + 1
        #     end_point = (
        #         start_point + my_instance.window_duration - 1
        #     )
        #     working_df_indiv = my_instance.df_indiv[
        #         (my_instance.
        #             df_indiv[it.tick_field] >= start_point) & (
        #             my_instance.df_indiv[it.tick_field] <= end_point)]

        # evaluate clustering
        # (dv, nb_clust_changes_loop,
        #     clustering_dual_values) = eval_clustering(
        #     my_instance, working_df_indiv,
        #     w, u, clustering_dual_values, constraints_dual,
        #     tol_clust, tol_move_clust,
        #     df_clust, cluster_profiles, labels_)

        # n_iter = n_iter - 1

        # First placement part
        if config['placement']['enable']:
            logging.info('Performing placement ... \n')
            start = time.time()
            place.allocation_distant_pairwise(
                my_instance, cluster_var_matrix, labels_)
            print('Placement heuristic time : %f s' % (time.time() - start))
            add_time(-1, 'placement_0', (time.time() - start))
        else:
            logging.info('We do not perform placement \n')
    elif method == 'spread':
        place.allocation_spread(my_instance, my_instance.nb_nodes)
    elif method == 'iter-consol':
        place.allocation_spread(my_instance)
    df_host_evo = my_instance.df_host.loc[
        my_instance.df_host[it.tick_field] <= my_instance.sep_time
    ]
    return (my_instance, df_host_evo,
            df_indiv_clust, labels_)


def run_period(
    my_instance: Instance, df_host_evo: pd.DataFrame,
    df_indiv_clust: pd.DataFrame, labels_: List,
    config: Dict, output_path: str, method: str, cluster_method: str
):
    """Perform all needed process during evaluation period."""
    nb_overloads = 0
    # Loops for evaluation
    if method in ['loop']:
        # loop 'streaming' progress
        it.results_file.write('\n### Loop process ###\n')
        (fig_node, fig_clust, fig_mean_clust,
         df_host_evo, nb_overloads) = streaming_eval(
            my_instance, df_indiv_clust, labels_,
            config['loop']['mode'],
            config['loop']['tick'],
            config['loop']['constraints_dual'],
            config['loop']['tol_dual_clust'],
            config['loop']['tol_move_clust'],
            config['loop']['tol_open_clust'],
            config['loop']['tol_dual_place'],
            config['loop']['tol_move_place'],
            config['loop']['tol_step'],
            cluster_method,
            df_host_evo)
        fig_node.savefig(output_path + '/node_evo_plot.svg')
        fig_clust.savefig(output_path + '/clust_evo_plot.svg')
        fig_mean_clust.savefig(output_path + '/mean_clust_evo_plot.svg')

    elif method in ['heur', 'spread', 'iter-consol']:
        # TODO adapt after 'progress_time_noloop' changed
        (temp_df_host, nb_overloads, _, _, _) = progress_time_noloop(
            my_instance, 'local',
            my_instance.sep_time, my_instance.df_indiv[it.tick_field].max(),
            labels_,
            0, config['loop']['constraints_dual'], {}, {},
            config['loop']['tol_dual_clust'],
            config['loop']['tol_move_clust'],
            config['loop']['tol_dual_place'],
            config['loop']['tol_move_place']
        )
        df_host_evo = df_host_evo.append(
            temp_df_host[~temp_df_host[it.tick_field].isin(
                df_host_evo[it.tick_field].unique())], ignore_index=True)

    return (df_host_evo, nb_overloads)


def streaming_eval(my_instance: Instance, df_indiv_clust: pd.DataFrame,
                   labels_: List, mode: str, tick: int,
                   constraints_dual: List,
                   tol_clust: float, tol_move_clust: float, tol_open_clust: float,
                   tol_place: float, tol_move_place: float, tol_step: float,
                   cluster_method: str, df_host_evo: pd.DataFrame
                   ) -> Tuple[plt.Figure, plt.Figure, plt.Figure, List, pd.DataFrame, int]:
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

    tmin = my_instance.sep_time - (my_instance.window_duration - 1)
    tmax = my_instance.sep_time

    total_loop_time = 0.0
    loop_nb = 1
    nb_clust_changes = 0
    nb_place_changes = 0
    total_nb_overload = 0
    end = False

    it.results_file.write('Loop mode : %s\n' % mode)
    logging.info('Beginning the loop process ...\n')

    (working_df_indiv, df_clust, w, u, v) = build_matrices(
        my_instance, tmin, tmax, labels_
    )

    start = time.time()
    (clust_model, place_model,
     clustering_dual_values, placement_dual_values) = pre_loop(
        my_instance, working_df_indiv, df_clust,
        w, u, constraints_dual, v, cluster_method
    )
    add_time(0, 'total_loop', (time.time() - start))

    tmin = my_instance.sep_time
    if mode == 'event':
        tmax = my_instance.df_indiv[it.tick_field].max()
    else:
        tmax += tick

    # TODO improve model builds
    while not end:
        loop_time = time.time()
        logging.info('\n # Enter loop number %d #\n' % loop_nb)
        it.results_file.write('\n # Loop number %d #\n' % loop_nb)
        it.optim_file.write('\n # Enter loop number %d #\n' % loop_nb)
        print('\n # Enter loop number %d #\n' % loop_nb)

        # TODO not fully tested (replace containers)
        (temp_df_host, nb_overload, loop_nb,
         nb_clust_changes_loop, nb_place_changes_loop) = progress_time_noloop(
            my_instance, 'local', tmin, tmax, labels_, loop_nb,
            constraints_dual, clustering_dual_values, placement_dual_values,
            tol_clust, tol_move_clust, tol_place, tol_move_place)
        df_host_evo = df_host_evo.append(
            temp_df_host[~temp_df_host[it.tick_field].isin(
                df_host_evo[it.tick_field].unique())], ignore_index=True)
        total_nb_overload += nb_overload
        nb_clust_changes += nb_clust_changes_loop
        nb_place_changes += nb_place_changes_loop

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

        start = time.time()
        (working_df_indiv, df_clust, w, u, v) = build_matrices(
            my_instance, tmin, tmax, labels_
        )
        add_time(loop_nb, 'build_matrices', (time.time() - start))
        nb_clust_changes_loop = 0
        nb_place_changes_loop = 0

        (init_loop_obj_nodes, init_loop_obj_delta) = mdl.get_obj_value_indivs(
            working_df_indiv)

        # TODO not very practical
        # plot.plot_clustering_containers_by_node(
        #     working_df_indiv, my_instance.dict_id_c, labels_)

        (nb_clust_changes_loop, nb_place_changes_loop,
            init_loop_silhouette, end_loop_silhouette,
            clust_conf_nodes, clust_conf_edges, clust_max_deg, clust_mean_deg,
            place_conf_nodes, place_conf_edges, place_max_deg, place_mean_deg,
            clust_model, place_model,
            clustering_dual_values, placement_dual_values,
            df_clust, cluster_profiles, labels_) = eval_sols(
            my_instance, working_df_indiv, cluster_method,
            w, u, v, clust_model, place_model,
            constraints_dual, clustering_dual_values, placement_dual_values,
            tol_clust, tol_move_clust, tol_open_clust, tol_place, tol_move_place,
            df_clust, cluster_profiles, labels_, loop_nb
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
        (end_loop_obj_nodes, end_loop_obj_delta) = mdl.get_obj_value_host(
            working_df_host)
        it.results_file.write('Loop delta before changes : %f\n' % init_loop_obj_delta)
        it.results_file.write('Loop delta after changes : %f\n' % end_loop_obj_delta)
        it.results_file.write('Loop time : %f s\n' % loop_time)
        add_time(loop_nb, 'total_loop', loop_time)
        total_loop_time += loop_time

        # TODO append deprecated
        # Save loop indicators in df
        it.loop_results = it.loop_results.append({
            'num_loop': int(loop_nb),
            'init_silhouette': init_loop_silhouette,
            'init_delta': init_loop_obj_delta,
            'clust_conf_nodes': clust_conf_nodes,
            'clust_conf_edges': clust_conf_edges,
            'clust_max_deg': clust_max_deg,
            'clust_mean_deg': clust_mean_deg,
            'clust_changes': int(nb_clust_changes_loop),
            'place_conf_nodes': place_conf_nodes,
            'place_conf_edges': place_conf_edges,
            'place_max_deg': place_max_deg,
            'place_mean_deg': place_mean_deg,
            'place_changes': int(nb_place_changes_loop),
            'end_silhouette': end_loop_silhouette,
            'end_delta': end_loop_obj_delta,
            'loop_time': loop_time
        }, ignore_index=True)

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
            df_host_evo, total_nb_overload)
    # return (fig_node, fig_clust, fig_mean_clust,
    #         [nb_clust_changes, nb_place_changes, total_loop_time, total_loop_time / loop_nb],
    #         df_host_evo, total_nb_overload)


def progress_time_noloop(
        instance: Instance, fixing: str, tmin: int, tmax: int, labels_, loop_nb,
        constraints_dual, clustering_dual_values, placement_dual_values,
        tol_clust, tol_move_clust, tol_place, tol_move_place) -> Tuple[pd.DataFrame, int]:
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
                     df_indiv[it.tick_field] >= tick - instance.window_duration) & (
                        instance.df_indiv[it.tick_field] <= tick)]
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


def pre_loop(
    my_instance: Instance, working_df_indiv: pd.DataFrame,
    df_clust: pd.DataFrame, w: np.array, u: np.array,
    constraints_dual: List, v: np.array, cluster_method: str
):
    """Build optimization problems and solve them with T_init solutions."""
    logging.info('Evaluation of problems with initial solutions')
    print('Building clustering model ...')
    start = time.time()
    clust_model = mdl.Model(1,
                               working_df_indiv,
                               my_instance.dict_id_c,
                               nb_clusters=my_instance.nb_clusters,
                               w=w, sol_u=u)
    clust_model.write_infile()
    add_time(0, 'build_clustering_model', (time.time() - start))
    start = time.time()
    print('Solving first clustering ...')
    logging.info('# Clustering evaluation #')
    logging.info('Solving linear relaxation ...')
    # clust_model.solve(clust_model.relax_mdl)
    add_time(0, 'solve_clustering_model', (time.time() - start))
    logging.info('Clustering problem not evaluated yet\n')
    print('\n ## Pyomo solve ## \n\n')
    clust_model.solve()
    clustering_dual_values = mdl.fill_dual_values(clust_model)

    if cluster_method == 'stream-km':
        it.streamkm_model = Streamkm(
            coresetsize=my_instance.nb_containers,
            length=my_instance.time * my_instance.nb_containers,
            seed=my_instance.nb_clusters,
        )
        it.streamkm_model.partial_fit(df_clust.drop('cluster', axis=1))
        _, labels_ = it.streamkm_model.get_final_clusters(
            my_instance.nb_clusters, seed=my_instance.nb_clusters)
        df_clust['cluster'] = labels_

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
    print('Building placement model ...')
    start = time.time()
    place_model = mdl.Model(2,
                               working_df_indiv,
                               my_instance.dict_id_c,
                               my_instance.dict_id_n,
                               my_instance.df_host_meta,
                               dv=dv, sol_u=u, sol_v=v)
    add_time(0, 'build_placement_model', (time.time() - start))
    start = time.time()
    print('Solving first placement ...')
    # place_model.solve(place_model.relax_mdl)
    logging.info('Placement problem not evaluated yet\n')
    add_time(0, 'solve_placement_model', (time.time() - start))
    print('\n ## Pyomo solve ## \n\n')
    place_model.solve()
    placement_dual_values = mdl.fill_dual_values(place_model)

    return (clust_model, place_model,
            clustering_dual_values, placement_dual_values)


def build_matrices(
    my_instance: Instance, tmin: int, tmax: int, labels_: List
):
    """Build period dataframe and matrices to be used."""
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

    return (working_df_indiv, df_clust, w, u, v)


def eval_sols(
        my_instance: Instance, working_df_indiv: pd.DataFrame,
        cluster_method: str, w: np.array, u, v, clust_model, place_model,
        constraints_dual, clustering_dual_values, placement_dual_values,
        tol_clust, tol_move_clust, tol_open_clust, tol_place, tol_move_place,
        df_clust, cluster_profiles, labels_, loop_nb
):
    """Evaluate clustering and placement solutions."""
    # evaluate clustering
    start = time.time()
    it.clustering_file.write(
        'labels before change\n'
    )
    it.clustering_file.write(
        np.array2string(labels_, separator=',')
    )
    it.clustering_file.write('\n')
    if cluster_method == 'loop-cluster':
        (dv, nb_clust_changes_loop,
            init_loop_silhouette, end_loop_silhouette,
            clustering_dual_values, clust_model,
            clust_conf_nodes, clust_conf_edges,
            clust_max_deg, clust_mean_deg) = eval_clustering(
            my_instance, w, u,
            clust_model, clustering_dual_values, constraints_dual,
            tol_clust, tol_move_clust, tol_open_clust,
            df_clust, cluster_profiles, labels_, loop_nb)
    elif cluster_method == 'kmeans-scratch':
        (dv, nb_clust_changes_loop,
            init_loop_silhouette, end_loop_silhouette,
            clustering_dual_values, clust_model,
            clust_conf_nodes, clust_conf_edges,
            clust_max_deg, clust_mean_deg) = loop_kmeans(
                my_instance, df_clust, labels_
        )
    elif cluster_method == 'stream-km':
        (dv, nb_clust_changes_loop,
            init_loop_silhouette, end_loop_silhouette,
            clustering_dual_values, clust_model,
            clust_conf_nodes, clust_conf_edges,
            clust_max_deg, clust_mean_deg) = stream_km(
                my_instance, df_clust, labels_
        )
    it.clustering_file.write(
        'labels after change\n'
    )
    it.clustering_file.write(
        np.array2string(labels_, separator=',')
    )
    it.clustering_file.write('\n')
    add_time(loop_nb, 'loop-clustering', (time.time() - start))
    it.clustering_file.write(
        'Loop clustering time : %f s\n' % (time.time() - start))

    # evaluate placement
    start = time.time()
    (nb_place_changes_loop,
        placement_dual_values, place_model,
        place_conf_nodes, place_conf_edges,
        place_max_deg, place_mean_deg) = eval_placement(
        my_instance, working_df_indiv,
        w, u, v, dv,
        placement_dual_values, constraints_dual, place_model,
        tol_place, tol_move_place, nb_clust_changes_loop, loop_nb
    )
    add_time(loop_nb, 'loop_placement', (time.time() - start))

    return (
        nb_clust_changes_loop, nb_place_changes_loop,
        init_loop_silhouette, end_loop_silhouette,
        clust_conf_nodes, clust_conf_edges,
        clust_max_deg, clust_mean_deg,
        place_conf_nodes, place_conf_edges,
        place_max_deg, place_mean_deg,
        clust_model, place_model,
        clustering_dual_values, placement_dual_values,
        df_clust, cluster_profiles, labels_
    )


def eval_clustering(my_instance: Instance,
                    w: np.array, u: np.array, clust_model,
                    clustering_dual_values: Dict, constraints_dual: Dict,
                    tol_clust: float, tol_move_clust: float, tol_open_clust: float,
                    df_clust: pd.DataFrame, cluster_profiles: np.array, labels_: List,
                    loop_nb) -> np.array:
    """Evaluate current clustering solution and update it if needed."""
    nb_clust_changes_loop = 0
    logging.info('# Clustering evaluation #')

    init_loop_silhouette = clt.get_silhouette(df_clust, labels_)

    start = time.time()
    clust_model.update_adjacency_clust_constraints(u)
    clust_model.update_obj_clustering(w)
    add_time(loop_nb, 'update_clustering_model', (time.time() - start))
    logging.info('Solving clustering linear relaxation ...')
    start = time.time()
    clust_model.solve()
    add_time(loop_nb, 'solve_clustering', (time.time() - start))
    # print('Init clustering lp solution : ', clust_model.relax_mdl.objective_value)

    logging.info('Checking for changes in clustering dual values ...')
    start = time.time()
    (moving_containers,
     clust_conflict_nodes,
     clust_conflict_edges,
     clust_max_deg, clust_mean_deg) = mdl.get_moving_containers_clust(
        clust_model, clustering_dual_values,
        tol_clust, tol_move_clust,
        my_instance.nb_containers, my_instance.dict_id_c,
        df_clust, cluster_profiles)
    add_time(loop_nb, 'get_moves_clustering', (time.time() - start))
    if len(moving_containers) > 0:
        logging.info('Changing clustering ...')
        start = time.time()
        ics = 0.0
        (df_clust, labels_, nb_clust_changes_loop) = clt.change_clustering(
            moving_containers, df_clust, labels_,
            my_instance.dict_id_c, tol_open_clust * ics)
        u = clt.build_adjacency_matrix(labels_)
        add_time(loop_nb, 'move_clustering', (time.time() - start))

        clust_model.update_adjacency_clust_constraints(u)
        logging.info('Solving linear relaxation after changes ...')
        start = time.time()
        clust_model.solve()
        clustering_dual_values = mdl.fill_dual_values(clust_model)
        add_time(loop_nb, 'solve_new_clustering', (time.time() - start))
    else:
        logging.info('Clustering seems still right ...')
        it.results_file.write('Clustering seems still right ...')

    # TODO improve this part (distance...)
    # TODO factorize
    # Compute new clusters profiles
    cluster_profiles = clt.get_cluster_mean_profile(
        df_clust)
    cluster_vars = clt.get_cluster_variance(cluster_profiles)
    cluster_var_matrix = clt.get_sum_cluster_variance(
        cluster_profiles, cluster_vars)
    dv = ctnr.build_var_delta_matrix_cluster(
        df_clust, cluster_var_matrix, my_instance.dict_id_c)

    end_loop_silhouette = clt.get_silhouette(df_clust, labels_)

    return (dv, nb_clust_changes_loop,
            init_loop_silhouette, end_loop_silhouette,
            clustering_dual_values, clust_model,
            clust_conflict_nodes, clust_conflict_edges,
            clust_max_deg, clust_mean_deg)


def eval_placement(my_instance: Instance, working_df_indiv: pd.DataFrame,
                   w: np.array, u: np.array, v: np.array, dv: np.array,
                   placement_dual_values: Dict, constraints_dual: Dict, place_model,
                   tol_place: float, tol_move_place: float,
                   nb_clust_changes_loop: int, loop_nb):
    """Evaluate current clustering solution and update it if needed."""
    logging.info('# Placement evaluation #')

    start = time.time()
    #TODO update without re-creating from scratch ? Study
    place_model = mdl.Model(2,
                               working_df_indiv,
                               my_instance.dict_id_c,
                               my_instance.dict_id_n,
                               my_instance.df_host_meta,
                               dv=dv, sol_u=u, sol_v=v)
    add_time(loop_nb, 'update_placement_model', (time.time() - start))
    it.optim_file.write('solve without any change\n')
    start = time.time()
    place_model.solve()
    add_time(loop_nb, 'solve_placement', (time.time() - start))
    # print('Init placement lp solution : ', place_model.relax_mdl.objective_value)
    moving_containers = []
    nb_place_changes_loop = 0
    place_conf_nodes = 0
    place_conf_edges = 0
    place_max_deg = 0
    place_mean_deg = 0

    if nb_clust_changes_loop > 0:
        logging.info('Checking for changes in placement dual values ...')
        start = time.time()
        (moving_containers,
         place_conf_nodes,
         place_conf_edges,
         place_max_deg, place_mean_deg) = mdl.get_moving_containers_place(
            place_model, placement_dual_values,
            tol_place, tol_move_place, my_instance.nb_containers,
            working_df_indiv, my_instance.dict_id_c)
        add_time(loop_nb, 'get_moves_placement', (time.time() - start))
        if len(moving_containers) > 0:
            nb_place_changes_loop = len(moving_containers)
            start = time.time()
            place.move_list_containers(moving_containers, my_instance,
                                       working_df_indiv[it.tick_field].min(),
                                       working_df_indiv[it.tick_field].max())
            add_time(loop_nb, 'move_placement', (time.time() - start))
            v = place.build_placement_adj_matrix(
                working_df_indiv, my_instance.dict_id_c)
            start = time.time()
            place_model.update_adjacency_place_constraints(v)
            place_model.update_obj_place(dv)
            place_model.solve()
            # print('After changes placement lp solution : ', place_model.relax_mdl.objective_value)
            placement_dual_values = mdl.fill_dual_values(place_model)
            add_time(loop_nb, 'solve_new_placement', (time.time() - start))
        else:
            logging.info('No container to move : we do nothing ...\n')

    return (nb_place_changes_loop, placement_dual_values, place_model,
            place_conf_nodes, place_conf_edges,
            place_max_deg, place_mean_deg)


def loop_kmeans(my_instance: Instance,
                df_clust: pd.DataFrame, labels_: List):
    """Update clustering via kmeans from scratch."""
    logging.info('# Clustering via k-means from scratch #')
    init_loop_silhouette = clt.get_silhouette(df_clust, labels_)

    new_labels_ = clt.perform_clustering(
        df_clust.drop('cluster', axis=1), 'kmeans', my_instance.nb_clusters)
    nb_clust_changes_loop = len(
        [i for i, j in zip(labels_, new_labels_) if i != j])
    labels_ = new_labels_
    df_clust['cluster'] = labels_

    # TODO factorize
    # Compute new clusters profiles
    cluster_profiles = clt.get_cluster_mean_profile(
        df_clust)
    cluster_vars = clt.get_cluster_variance(cluster_profiles)
    cluster_var_matrix = clt.get_sum_cluster_variance(
        cluster_profiles, cluster_vars)
    dv = ctnr.build_var_delta_matrix_cluster(
        df_clust, cluster_var_matrix, my_instance.dict_id_c)

    end_loop_silhouette = clt.get_silhouette(df_clust, labels_)

    return (dv, nb_clust_changes_loop,
            init_loop_silhouette, end_loop_silhouette,
            {}, None, 0, 0, 0, 0)


def stream_km(my_instance: Instance,
              df_clust: pd.DataFrame, labels_: List):
    """Update clustering via kmeans from scratch."""
    logging.info('# Clustering via streamkm #')
    init_loop_silhouette = clt.get_silhouette(df_clust, labels_)

    it.streamkm_model.partial_fit(df_clust.drop('cluster', axis=1))
    _, new_labels_ = it.streamkm_model.get_final_clusters(
        my_instance.nb_clusters, seed=my_instance.nb_clusters)

    nb_clust_changes_loop = len(
        [i for i, j in zip(labels_, new_labels_) if i != j])
    labels_ = new_labels_
    df_clust['cluster'] = labels_

    # TODO factorize
    # Compute new clusters profiles
    cluster_profiles = clt.get_cluster_mean_profile(
        df_clust)
    cluster_vars = clt.get_cluster_variance(cluster_profiles)
    cluster_var_matrix = clt.get_sum_cluster_variance(
        cluster_profiles, cluster_vars)
    dv = ctnr.build_var_delta_matrix_cluster(
        df_clust, cluster_var_matrix, my_instance.dict_id_c)

    end_loop_silhouette = clt.get_silhouette(df_clust, labels_)

    return (dv, nb_clust_changes_loop,
            init_loop_silhouette, end_loop_silhouette,
            {}, None, 0, 0, 0, 0)


def end_loop(working_df_indiv: pd.DataFrame, tmin: int,
             nb_clust_changes: int, nb_place_changes: int, nb_overload: int,
             total_loop_time: float, loop_nb: int, df_host_evo: pd.DataFrame):
    """Perform all stuffs after last loop."""
    working_df_host = working_df_indiv.groupby(
        [working_df_indiv[it.tick_field], it.host_field],
        as_index=False).agg(it.dict_agg_metrics)

    (end_loop_obj_nodes, end_loop_obj_delta) = mdl.get_obj_value_host(
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


def add_time(loop_nb: int, action: str, time: float):
    """Add an action time in times dataframe."""
    it.times_df = it.times_df.append({
        'num_loop': loop_nb,
        'action': action,
        'time': time
    }, ignore_index=True)


def close_files():
    """Write the final files and close all open files."""
    it.results_file.close()


if __name__ == '__main__':
    main()
