"""
Entry point of hots module through ``hots path [OPTIONS]``.

    - path is the folder where we find the files

        * container_usage.csv : describes container resource consumption
        * node_meta.csv : describes nodes capacities
        * (node_usage.csv : describes nodes resource consumption)

    - type ``hots --help`` for options description

The entire methodology is called from here (initialization, clustering,
allocation, evaluation, access to optimization model...).
"""

import logging
import math
import os
import signal
import time

import click

from clusopt_core.cluster import Streamkm

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import psutil

from . import clustering as clt
from . import container as ctnr
from . import init as it
from . import model as mdl
from . import node
from . import placement as place
from . import plot
from . import reader
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
@click.option('-C', '--tolclust', required=False, type=str,
              help='Use specific value for epsilonC (clustering conflict threshold)')
@click.option('-A', '--tolplace', required=False, type=str,
              help='Use specific value for epsilonA (placement conflict threshold)')
@click.option('-K', '--use_kafka', required=False, type=bool, default=False,
              help='Use Kafka streaming platform for data processing')
def main(path, k, tau, method, cluster_method, param, output, tolclust, tolplace, use_kafka):
    """Use method to propose a placement solution for micro-services adjusted in time.

    :param path: path to folder with data and parameters file
    :type path: click.Path
    :param k: number of clusters to use for clustering
    :type k: int
    :param tau: time window size for the loop
    :type tau: int
    :param method: method to use to solve initial problem
    :type method: str
    :param cluster_method: method to use to update clustering
    :type cluster_method: str
    :param param: parameter file to use
    :type param: str
    :param output: output folder to use
    :type output: str
    :param tolclust: threshold to use for clustering conflict
    :type tolclust: str
    :param tolplace: threshold to use for placement conflict
    :type tolplace: str
    :param use_kafka: streaming platform
    :type use_kafka: bool
    """
    # Initialization part
    main_time = time.time()
    tot_mem_before = process_memory()
    start_time = time.time()

    signal.signal(signal.SIGINT, signal_handler_sigint)
    if not path[-1] == '/':
        path += '/'

    start = time.time()
    (config, output_path, my_instance) = preprocess(
        path, k, tau, method, cluster_method, param, output, tolclust, tolplace, use_kafka
    )
    add_time(-1, 'preprocess', (time.time() - start))
    mem_before = my_instance.df_indiv.memory_usage(index=True).sum()

    # Plot initial data
    # TODO remove ? better option to performe ?
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

    # Global loop for getting data
    print(it.csv_reader)
    print(it.avro_deserializer)
    # it.s_entry = True
    current_time = 0
    # Offline / online separation : TODO in parameters
    offline_sep = 3
    my_instance.sep_time = offline_sep
    tick = config['loop']['tick']

    total_loop_time = 0.0
    loop_nb = 1
    nb_clust_changes = 0
    nb_place_changes = 0
    total_nb_overload = 0

    print('Ready for new data...')
    try:
        while it.s_entry:
            if current_time < offline_sep:
                current_data = reader.get_next_data(
                    current_time, offline_sep, offline_sep + 1, use_kafka
                )
                current_time += offline_sep
                my_instance.df_indiv = current_data
                my_instance.df_host = current_data.groupby(
                    [current_data[it.tick_field], current_data[it.host_field]],
                    as_index=False).agg(it.dict_agg_metrics)

                # Analysis period
                start = time.time()
                (my_instance, df_host_evo,
                 df_indiv_clust, labels_) = analysis_period(
                    my_instance, config, method
                )
                add_time(-1, 'total_t_obs', (time.time() - start))

                cluster_profiles = clt.get_cluster_mean_profile(df_indiv_clust)
                tmin = my_instance.sep_time - (my_instance.window_duration - 1)
                tmax = my_instance.sep_time

                # it.results_file.write('Loop mode : %s\n' % mode)
                logging.info('Beginning the loop process ...\n')

                (working_df_indiv, df_clust, w, u, v) = build_matrices(
                    my_instance, tmin, tmax, labels_
                )

                # build initial optimisation model in pre loop using offline data
                start = time.time()
                (clust_model, place_model,
                 clustering_dual_values, placement_dual_values) = pre_loop(
                    my_instance, working_df_indiv, df_clust,
                    w, u, config['loop']['constraints_dual'], v,
                    cluster_method, config['optimization']['solver']
                )
                add_time(0, 'total_loop', (time.time() - start))
                print('ready for loop ?')
                tmax += tick
                tmin = tmax - (my_instance.window_duration - 1)

            else:
                last_index = my_instance.df_indiv.index.levels[0][-1]
                current_data = reader.get_next_data(
                    current_time, config['loop']['tick'],
                    config['loop']['tick'] - current_time + 1, use_kafka
                )
                current_time += config['loop']['tick']
                my_instance.df_indiv = pd.concat(
                    [my_instance.df_indiv, current_data])
                new_df_host = current_data.groupby(
                    [current_data[it.tick_field], it.host_field], as_index=False
                ).agg(it.dict_agg_metrics)
                new_df_host = new_df_host.astype({
                    it.host_field: str,
                    it.tick_field: int}
                )
                previous_timestamp = last_index
                existing_machine_ids = my_instance.df_host[
                    my_instance.df_host[it.tick_field] == previous_timestamp
                ][it.host_field].unique()
                missing_machine_ids = set(existing_machine_ids) - set(
                    new_df_host[it.host_field])

                missing_rows = pd.DataFrame({
                    'timestamp': int(current_time),
                    'machine_id': list(missing_machine_ids),
                    'cpu': 0.0
                })
                # new_df_host.sort_values(it.tick_field, inplace=True)
                new_df_host = pd.concat([new_df_host, missing_rows])
                new_df_host.set_index([it.tick_field, it.host_field], inplace=True, drop=False)
                my_instance.df_host = pd.concat([
                    my_instance.df_host, new_df_host
                ])
                print(current_data)
                print(current_time)
                print(my_instance.df_indiv)
                print('perform loop')
                (working_df_indiv, df_clust, w, u, v) = build_matrices(
                    my_instance, tmin, tmax, labels_
                )
                nb_clust_changes_loop = 0
                nb_place_changes_loop = 0
                (nb_clust_changes_loop, nb_place_changes_loop,
                    init_loop_silhouette, end_loop_silhouette,
                    clust_conf_nodes, clust_conf_edges, clust_max_deg, clust_mean_deg,
                    place_conf_nodes, place_conf_edges, place_max_deg, place_mean_deg,
                    clust_model, place_model,
                    clustering_dual_values, placement_dual_values,
                    df_clust, cluster_profiles, labels_) = eval_sols(
                    my_instance, working_df_indiv, cluster_method,
                    w, u, v, clust_model, place_model,
                    config['loop']['constraints_dual'],
                    clustering_dual_values, placement_dual_values,
                    config['loop']['tol_dual_clust'],
                    config['loop']['tol_move_clust'],
                    config['loop']['tol_open_clust'],
                    config['loop']['tol_dual_place'],
                    config['loop']['tol_move_place'],                                   
                    df_clust, cluster_profiles, labels_, loop_nb,
                    config['optimization']['solver']
                )
                it.results_file.write(
                    'Number of changes in clustering : %d\n' % nb_clust_changes_loop
                )
                it.results_file.write(
                    'Number of changes in placement : %d\n' % nb_place_changes_loop
                )
                nb_clust_changes += nb_clust_changes_loop
                nb_place_changes += nb_place_changes_loop
                tmax += tick
                tmin = tmax - (my_instance.window_duration - 1)
                my_instance.time += 1
                loop_nb += 1
                print('success ?')
            input()

    finally:
        # Close down consumer to commit final offsets.
        reader.close_reader(use_kafka)

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
        config, output_path, method, cluster_method, use_kafka
    )
    add_time(-1, 'total_t_run', (time.time() - start))
    total_method_time = time.time() - total_method_time
    # print("final df_host_evo: ",df_host_evo)
    # Print objectives of evaluation part
    (obj_nodes, obj_delta) = mdl.get_obj_value_host(df_host_evo)
    it.results_file.write('Number of nodes : %d, Ampli max : %f\n' % (
        obj_nodes, obj_delta))
    it.results_file.write('Number of overloads : %d\n' % nb_overloads)
    it.results_file.write('Total execution time : %f s\n\n' % (total_method_time))

    it.clustering_file.write('\nFinal k : %d' % my_instance.nb_clusters)

    # # Save evaluation results in files
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
    tot_mem_after = process_memory()
    mem_after = my_instance.df_indiv.memory_usage(index=True).sum()
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print('Elapsed time:', elapsed_time, 'seconds')
    print('memory use: ', tot_mem_after - tot_mem_before)
    print('dataframe memory use: ', mem_before)
    print('dataframe memory use: ', mem_after)
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
    path, k, tau, method, cluster_method, param, output, tolclust, tolplace, use_kafka
):
    """Load configuration, data and initialize needed objects.

    :param path: initial folder path
    :type path: str
    :param k: number of clusters to use for clustering
    :type k: int
    :param tau: time window size for loop
    :type tau: int
    :param method: method to use to solve initial problem
    :type method: str
    :param cluster_method: method to use to update clustering
    :type cluster_method: str
    :param param: parameter file to use
    :type param: str
    :param output: output folder to use
    :type output: str
    :param tolclust: threshold to use for clustering conflict
    :type tolclust: float
    :param tolplace: threshold to use for placement conflict
    :type tolplace: float
    :param use_kafka: streaming platform
    :type use_kafka: bool
    :return: tuple with needed parameters, path to output and Instance object
    :rtype: Tuple[Dict, str, Instance]
    """
    # TODO what if we want tick < tau ?
    print('Preprocessing ...')
    (config, output_path) = it.read_params(
        path, k, tau, method, cluster_method, param, output, use_kafka)
    config = spec_params(config, [tolclust, tolplace])
    logging.basicConfig(filename=output_path + '/logs.log', filemode='w',
                        format='%(message)s', level=logging.INFO)
    plt.style.use('bmh')

    # Init containers & nodes data, then Instance
    logging.info('Loading data and creating Instance (Instance information are in results file)\n')
    if use_kafka:
        reader.consume_all_data(config)
        # reader.delete_kafka_topic(config)
        reader.csv_to_stream(path, config)
    reader.init_reader(path, use_kafka)
    instance = Instance(path, config)
    it.results_file.write('Method used : %s\n' % method)
    instance.print_times(config['loop']['tick'])

    return (config, output_path, instance)


# TODO use Dict instead of List for genericity ?
def spec_params(config, list_params):
    """Define specific parameters.

    :param config: initial parameters set
    :type config: Dict
    :param list_params: parameters list for threshold
    :type list_params: List
    :return: final parameters set
    :rtype: Dict
    """
    if list_params[0] is not None:
        config['loop']['tol_dual_clust'] = float(list_params[0])
    if list_params[1] is not None:
        config['loop']['tol_dual_place'] = float(list_params[1])
    return config


def analysis_period(my_instance, config, method):
    """Perform all needed process during analysis period (T_init).

    :param my_instance: Instance object
    :type my_instance: Instance
    :param config: parameters set
    :type config: Dict
    :param method: method used to solve initial problem
    :type method: str
    :return: tuple with Instance object, nodes data with new placement, clustering data and
        clustering labels
    :rtype: Tuple
    """
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
    my_instance, df_host_evo, df_indiv_clust, labels_, config, output_path,
    method, cluster_method, use_kafka
):
    """Perform all needed process during evaluation period.

    :param my_instance: Instance object
    :type my_instance: Instance
    :param df_host_evo: evolving node data
    :type df_host_evo: pd.DataFrame
    :param df_indiv_clust: clustering individuals data
    :type df_indiv_clust: pd.DataFrame
    :param labels_: clustering labels
    :type labels_: List
    :param config: set parameters
    :type config: Dict
    :param output_path: output folder path
    :type output_path: str
    :param method: method to use to solve problem
    :type method: str
    :param cluster_method: method to use for clustering
    :type cluster_method: str
    :param use_kafka: streaming platform
    :type use_kafka: bool
    :return: evolving node data and number of nodes overloads
    :rtype: Tuple[pd.DataFrame, int]
    """
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
            config['optimization']['solver'],
            df_host_evo, use_kafka)
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
        df_host_evo = pd.concat([
            df_host_evo,
            temp_df_host[~temp_df_host[it.tick_field].isin(
                df_host_evo[it.tick_field].unique())]
        ])

    return (df_host_evo, nb_overloads)


def signal_handler_sigint(signal_number, frame):
    """Handle for exiting application via signal."""
    print('Exit application')
    it.s_entry = False


def streaming_eval(
    my_instance, df_indiv_clust, labels_, mode, tick, constraints_dual,
    tol_clust, tol_move_clust, tol_open_clust, tol_place, tol_move_place, tol_step,
    cluster_method, solver, df_host_evo, use_kafka
):
    """Define the streaming process for evaluation.

    :param my_instance: Instance object
    :type my_instance: Instance
    :param df_indiv_clust: clustering individual data
    :type df_indiv_clust: pd.DataFrame
    :param labels_: clustering labels
    :type labels_: List
    :param mode: mode used to trigger the loop
    :type mode: str
    :param tick: number of new datapoints before new loop
    :type tick: int
    :param constraints_dual: constraints type compared to trigger conflicts
    :type constraints_dual: List
    :param tol_clust: threshold used for clustering conflicts
    :type tol_clust: float
    :param tol_move_clust: threshold used for number of clustering moves
    :type tol_move_clust: float
    :param tol_open_clust: threshold used for opening new cluster
    :type tol_open_clust: float
    :param tol_place: threshold used for placement conflicts
    :type tol_place: float
    :param tol_move_place: threshold used for number of placement moves
    :type tol_move_place: float
    :param tol_step: step to increment previous threshold
    :type tol_step: float
    :param cluster_method: method used to update clustering
    :type cluster_method: str
    :param solver: solver used for pyomo
    :type solver: str
    :param df_host_evo: evolving node data
    :type df_host_evo: pd.DataFrame
    :param use_kafka: streaming platform
    :type use_kafka: bool
    :return: figures to plot to see clustering and nodes evolution + evolving node data + number
        of node overloads
    :rtype: Tuple[plt.Figure, plt.Figure, plt.Figure, List, pd.DataFrame, int]
    """
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

    it.results_file.write('Loop mode : %s\n' % mode)
    logging.info('Beginning the loop process ...\n')

    (working_df_indiv, df_clust, w, u, v) = build_matrices(
        my_instance, tmin, tmax, labels_
    )

    # build initial optimisation model in pre loop using all or potion of historical data.
    start = time.time()
    (clust_model, place_model,
     clustering_dual_values, placement_dual_values) = pre_loop(
        my_instance, working_df_indiv, df_clust,
        w, u, constraints_dual, v, cluster_method, solver
    )
    add_time(0, 'total_loop', (time.time() - start))

    if mode == 'event':
        tmin = my_instance.sep_time
        tmin = my_instance.sep_time
        tmax = my_instance.df_indiv[it.tick_field].max()
    else:
        tmax += tick
        tmin = tmax - (my_instance.window_duration - 1)

    if use_kafka:
        use_schema = False
        avro_deserializer = reader.connect_schema(
            use_schema, it.kafka_schema_url)
        # time_to_send = my_instance.df_indiv['timestamp'].iloc[-1]
        # history = True # consider historical data
        # send last historical data to kafka
        # reader.produce_data(my_instance, time_to_send, history)

        analysis_duration = 1  # perform optimization and placement when it equates the tick value
        it.time_at = []
        it.memory_usage = []
        it.tick_time = []
        it.total_mem_use = []
        # mem_before = my_instance.df_indiv.memory_usage(index=True).sum()
        # hist_time = list(set(my_instance.df_indiv['timestamp']))

        # for x in hist_time:
        #     it.time_at.append(x)
        #     it.memory_usage.append(mem_before)
        #     it.total_mem_use.append(tot_mem_after)
        it.s_entry = True
        # print('df_indiv1: ',my_instance.df_indiv)
        print('Ready for new data...')
        try:
            while it.s_entry:

                loop_time = time.time()
                it.kafka_consumer.subscribe([it.kafka_topics['docker_topic']])
                dval = reader.process_kafka_msg(avro_deserializer)

                if dval is None:
                    continue
                else:
                    key = list(dval.values())[0]
                    value = list(dval.values())[1]
                    file = list(dval.values())[2]
                    last_index = my_instance.df_indiv.index.levels[0][-1]
                    # print('last_index: ',last_index)
                    subset = my_instance.df_indiv[
                        my_instance.df_indiv.timestamp == last_index
                    ].copy()
                    subset.reset_index(drop=True, inplace=True)
                    subset.loc[:, 'timestamp'] = int(key)
                    subset.set_index([it.tick_field, it.indiv_field], inplace=True, drop=False)
                    subset.sort_index(inplace=True)

                    new_df_container = reassign_node(value)

                    new_df_container['machine_id'] = subset['machine_id'].where(
                        new_df_container['container_id'] == subset['container_id']
                    )
                    subset = subset.truncate(before=-1, after=-1)
                    # print('new_df_container: ',new_df_container)
                    logging.info('\n # Enter loop number %d #\n' % loop_nb)
                    it.results_file.write('\n # Loop number %d #\n' % loop_nb)
                    it.optim_file.write('\n # Enter loop number %d #\n' % loop_nb)

                    # TODO not fully tested (replace containers)

                    new_df_host = new_df_container.groupby(
                        [new_df_container[it.tick_field], it.host_field], as_index=False
                    ).agg(it.dict_agg_metrics)
                    new_df_host = new_df_host.astype({
                        it.host_field: str,
                        it.tick_field: int})
                    # new_df_host.sort_values(it.tick_field, inplace=True)
                    # Set remaining machine_ids from df_host to 0.0
                    previous_timestamp = last_index
                    existing_machine_ids = my_instance.df_host[
                        my_instance.df_host[it.tick_field] == previous_timestamp
                    ][it.host_field].unique()
                    missing_machine_ids = set(existing_machine_ids) - set(
                        new_df_host[it.host_field])

                    missing_rows = pd.DataFrame({
                        'timestamp': int(key),
                        'machine_id': list(missing_machine_ids),
                        'cpu': 0.0
                    })
                    # new_df_host.sort_values(it.tick_field, inplace=True)
                    new_df_host = pd.concat([new_df_host, missing_rows])
                    new_df_host.set_index([it.tick_field, it.host_field], inplace=True, drop=False)

                    # update df_indiv on every loop!
                    my_instance.df_indiv = pd.concat([
                        my_instance.df_indiv, new_df_container
                    ])
                    my_instance.df_host = pd.concat([
                        my_instance.df_host, new_df_host
                    ])

                    # print('\n # Step 1: Check Progress time no loop%d #\n' % loop_nb)
                    (temp_df_host, nb_overload, loop_nb,
                        nb_clust_changes_loop, nb_place_changes_loop) = progress_time_noloop(
                        my_instance, 'local', tmin, tmax, labels_, loop_nb,
                        constraints_dual, clustering_dual_values, placement_dual_values,
                        tol_clust, tol_move_clust, tol_place, tol_move_place, int(key))
                    df_host_evo = pd.concat([
                        df_host_evo,
                        temp_df_host[~temp_df_host[it.tick_field].isin(
                            df_host_evo[it.tick_field].unique())]
                    ])

                    total_nb_overload += nb_overload
                    nb_clust_changes += nb_clust_changes_loop
                    nb_place_changes += nb_place_changes_loop

                    if mode == 'event':
                        (temp_df_host, nb_overload, loop_nb,
                            nb_clust_changes, nb_place_changes) = progress_time_noloop(
                            my_instance, 'loop', tmin, tmax, labels_, loop_nb,
                            constraints_dual, clustering_dual_values, placement_dual_values,
                            tol_clust, tol_move_clust, tol_place, tol_move_place, int(key))
                        df_host_evo = pd.concat([
                            df_host_evo,
                            temp_df_host[~temp_df_host[it.tick_field].isin(
                                df_host_evo[it.tick_field].unique())]
                        ])
                        total_nb_overload += nb_overload

                    if analysis_duration == tick:  # equate to tick size
                        print('\n # Enter loop number %d #\n' % loop_nb)

                        # print('\n # Step 2: evaluate the solution at loop: %d #\n' % loop_nb)
                        analysis_duration = 1

                        start = time.time()
                        print('tmin, tmax: ', tmin, tmax)
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
                            df_clust, cluster_profiles, labels_, loop_nb, solver
                        )

                        it.results_file.write(
                            'Number of changes in clustering : %d\n' % nb_clust_changes_loop
                        )
                        it.results_file.write(
                            'Number of changes in placement : %d\n' % nb_place_changes_loop
                        )
                        nb_clust_changes += nb_clust_changes_loop
                        nb_place_changes += nb_place_changes_loop

                        # update clustering & node consumption plot
                        plot.update_clustering_plot(
                            fig_clust, ax_clust, df_clust, my_instance.dict_id_c)

                        plot.update_cluster_profiles(
                            fig_mean_clust, ax_mean_clust, cluster_profiles,
                            sorted(working_df_indiv[it.tick_field].unique())
                        )

                        plot.update_nodes_plot(
                            fig_node, ax_node, working_df_indiv, my_instance.dict_id_n)

                        working_df_indiv = my_instance.df_indiv[
                            (my_instance.
                                df_indiv[it.tick_field] >= tmin) & (
                                my_instance.df_indiv[it.tick_field] <= tmax)]
                        working_df_host = working_df_indiv.groupby(
                            [working_df_indiv[it.tick_field], it.host_field],
                            as_index=False).agg(it.dict_agg_metrics)

                        if loop_nb > 1:
                            df_host_evo = pd.concat(
                                [
                                    df_host_evo,
                                    working_df_host[~working_df_host[it.tick_field].isin(
                                        df_host_evo[it.tick_field].unique())]
                                ]
                            )

                        loop_time = (time.time() - loop_time)
                        (end_loop_obj_nodes, end_loop_obj_delta) = mdl.get_obj_value_host(
                            working_df_host)
                        it.results_file.write(
                            'Loop delta before changes : %f\n' % init_loop_obj_delta)
                        it.results_file.write(
                            'Loop delta after changes : %f\n' % end_loop_obj_delta)
                        it.results_file.write('Loop time : %f s\n' % loop_time)
                        add_time(loop_nb, 'total_loop', loop_time)
                        total_loop_time += loop_time

                # Save loop indicators in df
                it.loop_results = pd.concat(
                    [
                        it.loop_results,
                        pd.DataFrame.from_records([{
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
                        }])
                    ]
                )

                # input('\nPress any key to progress in time ...\n')
                tmax += tick
                tmin = tmax - (my_instance.window_duration - 1)
                # if tol_clust < 1.0:
                #     tol_clust += tol_step
                # if tol_place < 1.0:
                #     tol_place += tol_step

                # if tmax >= my_instance.time:

                #     it.s_entry = True  # change to False to end loop according to mock data
                # else:
                #     loop_nb += 1
                my_instance.time += 1
                loop_nb += 1
                # mem_after = process_memory()
                # memory_usage.append(mem_after - mem_before)
                mem_after = my_instance.df_indiv.memory_usage(index=True).sum()
                it.memory_usage.append(mem_after)
                it.time_at.append(key)
                if file:
                    break

        finally:
            # Close down consumer to commit final offsets.
            print('close kafka consumer')
            it.kafka_consumer.close()

    else:
        end = False
        # TODO improve model builds
        while not end:
            loop_time = time.time()
            logging.info('\n # Enter loop number %d #\n' % loop_nb)
            it.results_file.write('\n # Loop number %d #\n' % loop_nb)
            it.optim_file.write('\n # Enter loop number %d #\n' % loop_nb)
            print('\n # Enter loop number %d #\n' % loop_nb)

            # TODO not fully tested (replace containers)
            (temp_df_host, nb_overload, loop_nb,
                nb_clust_changes_loop, nb_place_changes_loop) = progress_time_noloop(
                my_instance, 'local', tmin, tmax, labels_, loop_nb,
                constraints_dual, clustering_dual_values, placement_dual_values,
                tol_clust, tol_move_clust, tol_place, tol_move_place, tick)
            df_host_evo = pd.concat([
                df_host_evo,
                temp_df_host[~temp_df_host[it.tick_field].isin(
                    df_host_evo[it.tick_field].unique())]
            ])
            total_nb_overload += nb_overload
            nb_clust_changes += nb_clust_changes_loop
            nb_place_changes += nb_place_changes_loop

            if mode == 'event':
                (temp_df_host, nb_overload, loop_nb,
                    nb_clust_changes, nb_place_changes) = progress_time_noloop(
                    my_instance, 'loop', tmin, tmax, labels_, loop_nb,
                    constraints_dual, clustering_dual_values, placement_dual_values,
                    tol_clust, tol_move_clust, tol_place, tol_move_place)
                df_host_evo = pd.concat([
                    df_host_evo,
                    temp_df_host[~temp_df_host[it.tick_field].isin(
                        df_host_evo[it.tick_field].unique())]
                ])
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
                df_clust, cluster_profiles, labels_, loop_nb, solver
            )

            it.results_file.write('Number of changes in clustering : %d\n' % nb_clust_changes_loop)
            it.results_file.write('Number of changes in placement : %d\n' % nb_place_changes_loop)
            nb_clust_changes += nb_clust_changes_loop
            nb_place_changes += nb_place_changes_loop

            # update clustering & node consumption plot
            plot.update_clustering_plot(
                fig_clust, ax_clust, df_clust, my_instance.dict_id_c)
            plot.update_cluster_profiles(
                fig_mean_clust, ax_mean_clust, cluster_profiles,
                sorted(working_df_indiv[it.tick_field].unique()))
            plot.update_nodes_plot(
                fig_node, ax_node, working_df_indiv, my_instance.dict_id_n)

            working_df_indiv = my_instance.df_indiv[
                (my_instance.
                    df_indiv[it.tick_field] >= tmin) & (
                    my_instance.df_indiv[it.tick_field] <= tmax)]
            working_df_host = working_df_indiv.groupby(
                [working_df_indiv[it.tick_field], it.host_field],
                as_index=False).agg(it.dict_agg_metrics)

            if loop_nb > 1:
                df_host_evo = pd.concat(
                    [
                        df_host_evo,
                        working_df_host[~working_df_host[it.tick_field].isin(
                            df_host_evo[it.tick_field].unique())]
                    ]
                )

            loop_time = (time.time() - loop_time)
            (end_loop_obj_nodes, end_loop_obj_delta) = mdl.get_obj_value_host(
                working_df_host)
            it.results_file.write('Loop delta before changes : %f\n' % init_loop_obj_delta)
            it.results_file.write('Loop delta after changes : %f\n' % end_loop_obj_delta)
            it.results_file.write('Loop time : %f s\n' % loop_time)
            add_time(loop_nb, 'total_loop', loop_time)
            total_loop_time += loop_time

            # Save loop indicators in df
            it.loop_results = pd.concat(
                [
                    it.loop_results,
                    pd.DataFrame.from_records([{
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
                    }])
                ]
            )

            # input('\nPress any key to progress in time ...\n')
            tmax += tick
            tmin = tmax - (my_instance.window_duration - 1)
            if tol_clust < 1.0:
                tol_clust += tol_step
            if tol_place < 1.0:
                tol_place += tol_step

            if tmax >= my_instance.time:
                end = True
            else:
                loop_nb += 1

    # plot.plot_memory_usage(it.time_at, it.total_mem_use, it.tick_time)
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
    instance, fixing, tmin, tmax, labels_, loop_nb,
    constraints_dual, clustering_dual_values, placement_dual_values,
    tol_clust, tol_move_clust, tol_place, tol_move_place, tick
):
    """We progress in time without performing the loop, checking node capacities.

    :param instance: Instance object
    :type instance: Instance
    :param fixing: method used to fix node assignment if overload
    :type fixing: str
    :param tmin: starting time of current window
    :type tmin: int
    :param tmax: final time of current window
    :type tmax: int
    :param labels_: clustering labels
    :type labels_: List
    :param loop_nb: current loop number
    :type loop_nb: int
    :param constraints_dual: constraints type compared to trigger conflicts
    :type constraints_dual: List
    :param clustering_dual_values: previous loop dual values for clustering
    :type clustering_dual_values: Dict
    :param placement_dual_values: previous loop dual values for placement
    :type placement_dual_values: Dict
    :param tol_clust: threshold used for clustering conflicts
    :type tol_clust: float
    :param tol_move_clust: threshold used for number of clustering moves
    :type tol_move_clust: float
    :param tol_place: threshold used for placement conflicts
    :type tol_place: float
    :param tol_move_place: threshold used for number of placement moves
    :type tol_move_place: float
    :param tick: current timestamp number
    :type tick: int
    :return: evolving node data + number of overloads + loop number + clustering and placement
        changes
    :rtype: Tuple[pd.DataFrame, int, int, int, int]
    """
    df_host_evo = pd.DataFrame(columns=instance.df_host.columns)
    nb_overload = 0
    nb_clust_changes = 0
    nb_place_changes = 0

    df_indiv = instance.df_indiv[instance.df_indiv[it.tick_field] == int(tick)].copy()
    # df_indiv = instance.df_indiv[instance.df_indiv[it.tick_field] >= tmin].copy()
    df_host_tick = df_indiv.groupby(
        [df_indiv[it.tick_field], it.host_field],
        as_index=False).agg(it.dict_agg_metrics)

    host_overload = node.check_capacities(df_host_tick, instance.df_host_meta)

    df_host_evo = pd.concat([
        df_host_evo, df_host_tick
    ])

    # print('TICK INFO',df_host_tick)  # timestamp machine_id cpu of node usage for window duration

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
    my_instance, working_df_indiv, df_clust, w, u, constraints_dual, v, cluster_method, solver
):
    """Build optimization problems and solve them with T_init solutions.

    :param my_instance: Instance object
    :type my_instance: Instance
    :param working_df_indiv: current loop data
    :type working_df_indiv: pd.DataFrame
    :param df_clust: clustering related data
    :type df_clust: pd.DataFrame
    :param w: dissimilarity matrix
    :type w: np.array
    :param u: clustering adjacency matrix
    :type u: np.array
    :param constraints_dual: constraints type compared to trigger conflicts
    :type constraints_dual: List
    :param v: placement adjacency matrix
    :type v: np.array
    :param cluster_method: method used to update clustering
    :type cluster_method: str
    :param solver: solver used for pyomo
    :type solver: str
    :return: optimization models and associated dual values
    :rtype: Tuple[mdl.Model, mdl.Model, Dict, Dict]
    """
    logging.info('Evaluation of problems with initial solutions')
    print('Building clustering model ...')
    start = time.time()
    clust_model = mdl.Model(
        1,
        working_df_indiv, it.metrics[0],
        my_instance.dict_id_c,
        nb_clusters=my_instance.nb_clusters,
        w=w, sol_u=u
    )
    add_time(0, 'build_clustering_model', (time.time() - start))
    start = time.time()
    print('Solving first clustering ...')
    logging.info('# Clustering evaluation #')
    logging.info('Solving linear relaxation ...')
    add_time(0, 'solve_clustering_model', (time.time() - start))
    logging.info('Clustering problem not evaluated yet\n')
    print('\n ## Pyomo solve ## \n\n')
    clust_model.solve(solver)
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
    place_model = mdl.Model(
        2,
        working_df_indiv, it.metrics[0],
        my_instance.dict_id_c,
        my_instance.dict_id_n,
        my_instance.df_host_meta,
        dv=dv, sol_u=u, sol_v=v
    )
    add_time(0, 'build_placement_model', (time.time() - start))
    start = time.time()
    print('Solving first placement ...')
    logging.info('Placement problem not evaluated yet\n')
    add_time(0, 'solve_placement_model', (time.time() - start))
    print('\n ## Pyomo solve ## \n\n')
    place_model.solve(solver)
    placement_dual_values = mdl.fill_dual_values(place_model)

    return (clust_model, place_model,
            clustering_dual_values, placement_dual_values)


def build_matrices(my_instance, tmin, tmax, labels_):
    """Build period dataframe and matrices to be used.

    :param my_instance: Instance object
    :type my_instance: Instance
    :param tmin: starting time of current window
    :type tmin: int
    :param tmax: final time of current window
    :type tmax: int
    :param labels_: clustering labels
    :type labels_: List
    :return: current loop data, clustering data, dissimilarity matrix and adjacency matrices
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, np.array, np.array, np.array]
    """
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
        my_instance, working_df_indiv,
        cluster_method, w, u, v, clust_model, place_model,
        constraints_dual, clustering_dual_values, placement_dual_values,
        tol_clust, tol_move_clust, tol_open_clust, tol_place, tol_move_place,
        df_clust, cluster_profiles, labels_, loop_nb, solver
):
    """Evaluate clustering and placement solutions.

    :param my_instance: Instance object
    :type my_instance: Instance
    :param working_df_indiv: current loop data
    :type working_df_indiv: pd.DataFrame
    :param cluster_method: method used to update clustering
    :type cluster_method: str
    :param w: dissimilarity matrix
    :type w: np.array
    :param u: clustering adjacency matrix
    :type u: np.array
    :param v: placement adjacency matrix
    :type v: np.array
    :param clust_model: clustering optimization model
    :type clust_model: mdl.Model
    :param place_model: placement optimization model
    :type place_model: mdl.Model
    :param constraints_dual: constraints type compared to trigger conflicts
    :type constraints_dual: List
    :param clustering_dual_values: previous loop dual values for clustering
    :type clustering_dual_values: Dict
    :param placement_dual_values: previous loop dual values for placement
    :type placement_dual_values: Dict
    :param tol_clust: threshold used for clustering conflicts
    :type tol_clust: float
    :param tol_move_clust: threshold used for number of clustering moves
    :type tol_move_clust: float
    :param tol_open_clust: threshold used for opening new cluster
    :type tol_open_clust: float
    :param tol_place: threshold used for placement conflicts
    :type tol_place: float
    :param tol_move_place: threshold used for number of placement moves
    :type tol_move_place: float
    :param df_clust: clustering related data
    :type df_clust: pd.DataFrame
    :param cluster_profiles: computed clusters mean profiles
    :type cluster_profiles: np.array
    :param labels_: cluster labels
    :type labels_: List
    :param loop_nb: current loop number
    :type loop_nb: int
    :param solver: solver used for pyomo
    :type solver: str
    :return: number of changes in clustering and placement, silhouette score before and after loop,
        number of nodes and edges in conflict graph + max and mean degree (clustering and
        placement), optimization models and associated dual values, clustering related data,
        clusters mean profiles and cluster labels
    :rtype: Tuple[
        int, int, float, float, int, int, float, float, int, int, float, float,
        mdl.Model, mdl.Model, Dict, Dict, pd.DataFrame, np.array, List
        ]
    """
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
            df_clust, cluster_profiles, labels_, loop_nb, solver)
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
        tol_place, tol_move_place, nb_clust_changes_loop, loop_nb, solver
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


# TODO update return type
def eval_clustering(
    my_instance, w, u, clust_model, clustering_dual_values, constraints_dual,
    tol_clust, tol_move_clust, tol_open_clust,
    df_clust, cluster_profiles, labels_, loop_nb, solver
):
    """Evaluate current clustering solution and update it if needed.

    :param my_instance: Instance object
    :type my_instance: Instance
    :param w: dissimilarity matrix
    :type w: np.array
    :param u: clustering adjacency matrix
    :type u: np.array
    :param clust_model: clustering optimization model
    :type clust_model: mdl.Model
    :param clustering_dual_values: previous loop dual values for clustering
    :type clustering_dual_values: Dict
    :param constraints_dual: constraints type compared to trigger conflicts
    :type constraints_dual: List
    :param tol_clust: threshold used for clustering conflicts
    :type tol_clust: float
    :param tol_move_clust: threshold used for number of clustering moves
    :type tol_move_clust: float
    :param tol_open_clust: threshold used for opening new cluster
    :type tol_open_clust: float
    :param df_clust: clustering related data
    :type df_clust: pd.DataFrame
    :param cluster_profiles: computed clusters mean profiles
    :type cluster_profiles: np.array
    :param labels_: cluster labels
    :type labels_: List
    :param loop_nb: current loop number
    :type loop_nb: int
    :param solver: solver used for pyomo
    :type solver: str
    :return: clustering variance matrix, number of changes in clustering, silhouette score before
        and after the loop, clustering dual values, clustering optimization model, number of nodes
        and edges in conflict graph + max and mean degree (clustering)
    :rtype: Tuple[
        np.array, int, float, float, Dict, mdl.Model, int, int, float, float
        ]
    """
    nb_clust_changes_loop = 0
    logging.info('# Clustering evaluation #')

    init_loop_silhouette = clt.get_silhouette(df_clust, labels_)

    start = time.time()
    clust_model.update_adjacency_clust_constraints(u)
    clust_model.update_obj_clustering(w)
    add_time(loop_nb, 'update_clustering_model', (time.time() - start))
    logging.info('Solving clustering linear relaxation ...')
    start = time.time()
    clust_model.solve(solver)
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
        clust_model.solve(solver)
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


def eval_placement(
    my_instance, working_df_indiv, w, u, v, dv,
    placement_dual_values, constraints_dual, place_model, tol_place, tol_move_place,
    nb_clust_changes_loop, loop_nb, solver
):
    """Evaluate current clustering solution and update it if needed.

    :param my_instance: Instance object
    :type my_instance: Instance
    :param working_df_indiv: current loop data
    :type working_df_indiv: pd.DataFrame
    :param w: dissimilarity matrix
    :type w: np.array
    :param u: clustering adjacency matrix
    :type u: np.array
    :param v: placement adjacency matrix
    :type v: np.array
    :param dv: clustering variance matrix
    :type dv: np.array
    :param placement_dual_values: previous loop dual values for placement
    :type placement_dual_values: Dict
    :param constraints_dual: constraints type compared to trigger conflicts
    :type constraints_dual: List
    :param place_model: placement optimization model
    :type place_model: mdl.Model
    :param tol_place: threshold used for placement conflicts
    :type tol_place: float
    :param tol_move_place: threshold used for number of placement moves
    :type tol_move_place: float
    :param nb_clust_changes_loop: number of changes in clustering in current loop
    :type nb_clust_changes_loop: int
    :param loop_nb: current loop number
    :type loop_nb: int
    :param solver: solver used for pyomo
    :type solver: str
    :return: number of placement changes, dual values related to current placement, placement
        optimization model, number of nodes and edges in conflict graph + max and mean degree
    :rtype: Tuple[int, Dict, mdl.Model, int, int, float, float]
    """
    logging.info('# Placement evaluation #')

    start = time.time()
    # TODO update without re-creating from scratch ? Study
    place_model = mdl.Model(
        2,
        working_df_indiv, it.metrics[0],
        my_instance.dict_id_c,
        my_instance.dict_id_n,
        my_instance.df_host_meta,
        dv=dv, sol_u=u, sol_v=v
    )
    add_time(loop_nb, 'update_placement_model', (time.time() - start))
    it.optim_file.write('solve without any change\n')
    start = time.time()
    place_model.solve(solver)
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
            place_model.solve(solver)
            # print('After changes placement lp solution : ', place_model.relax_mdl.objective_value)
            placement_dual_values = mdl.fill_dual_values(place_model)
            add_time(loop_nb, 'solve_new_placement', (time.time() - start))
        else:
            logging.info('No container to move : we do nothing ...\n')

    return (nb_place_changes_loop, placement_dual_values, place_model,
            place_conf_nodes, place_conf_edges,
            place_max_deg, place_mean_deg)


def loop_kmeans(my_instance, df_clust, labels_):
    """Update clustering via kmeans from scratch.

    :param my_instance: Instance object
    :type my_instance: Instance
    :param df_clust: clustering related data
    :type df_clust: pd.DataFrame
    :param labels_: clustering labels
    :type labels_: List
    :return: clustering variance matrix, number of changes in clustering, silhouette score before
        and after loop, null objects to match other return objects
    :rtype: Tuple[
        np.array, int, float, float, Dict, mdl.Model, int, int, float, float
        ]
    """
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


def stream_km(my_instance, df_clust, labels_):
    """Update clustering via kmeans from scratch.

    :param my_instance: Instance object
    :type my_instance: Instance
    :param df_clust: clustering related data
    :type df_clust: pd.DataFrame
    :param labels_: clustering labels
    :type labels_: List
    :return: clustering variance matrix, number of changes in clustering, silhouette score before
        and after loop, null objects to match other return objects
    :rtype: Tuple[
        np.array, int, float, float, Dict, mdl.Model, int, int, float, float
        ]
    """
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


def end_loop(
    working_df_indiv, tmin, nb_clust_changes, nb_place_changes, nb_overload,
    total_loop_time, loop_nb, df_host_evo
):
    """Perform all stuffs after last loop.

    :param working_df_indiv: current loop data
    :type working_df_indiv: pd.DataFrame
    :param tmin: starting time of current window
    :type tmin: int
    :param nb_clust_changes: number of changes in clustering
    :type nb_clust_changes: int
    :param nb_place_changes: number of changes in placement
    :type nb_place_changes: int
    :param nb_overload: number of node overload
    :type nb_overload: int
    :param total_loop_time: total running time for all loops
    :type total_loop_time: float
    :param loop_nb: final loop number
    :type loop_nb: int
    :param df_host_evo: evolving node data
    :type df_host_evo: pd.DataFrame
    :return: evolving node data
    :rtype: pd.DataFrame
    """
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
        df_host_evo = pd.concat(
            [
                df_host_evo,
                working_df_host[~working_df_host[it.tick_field].isin(
                    df_host_evo[it.tick_field].unique())]
            ]
        )
    return df_host_evo


def add_time(loop_nb, action, time):
    """Add an action time in times dataframe.

    :param loop_nb: current loop number
    :type loop_nb: int
    :param action: action or method we add time for
    :type action: str
    :param time: running time of the action
    :type time: float
    """
    it.times_df = pd.concat(
        [
            it.times_df,
            pd.DataFrame.from_records([{
                'num_loop': loop_nb,
                'action': action,
                'time': time
            }])
        ]
    )


def close_files():
    """Write the final files and close all open files."""
    it.results_file.close()


def process_memory():
    """Get memory information."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_mb = mem_info.rss / 1024 / 1024
    return mem_usage_mb


def reassign_node(c_info):
    """Reassign node in containers df."""
    # c_info = value['containers']
    new_df_container = pd.DataFrame(c_info)
    new_df_container = new_df_container.astype({
        it.indiv_field: str,
        it.host_field: str,
        it.tick_field: int})
    new_df_container.sort_values(it.tick_field, inplace=True)
    new_df_container.set_index([it.tick_field, it.indiv_field], inplace=True, drop=False)
    new_df_container.sort_index(inplace=True)
    return new_df_container


if __name__ == '__main__':
    main()
