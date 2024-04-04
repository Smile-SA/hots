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

import json
import logging
import math
import signal
import sys
import time

import click

from clusopt_core.cluster import Streamkm

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

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
@click.argument('config_path', required=True, type=click.Path(exists=True))
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
@click.option('-T', '--time_limit', required=False, type=int,
              help='Provide a time limit for data processing (in seconds)')
def main(config_path, k, tau, method, cluster_method, param, output, tolclust, tolplace, use_kafka, time_limit):
    """Use method to propose a placement solution for micro-services adjusted in time.

    :param config_path: path to config file with all information
    :type config_path: click.Path
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
    end_time = 0
    if time_limit is not None:
        end_time = main_time + time_limit * 60

    signal.signal(signal.SIGINT, signal_handler_sigint)
    signal.signal(signal.SIGTSTP, signal_handler_sigtstp)

    start = time.time()
    (config, output_path) = preprocess(
        config_path, k, tau, method, cluster_method, param, output, tolclust, tolplace, use_kafka
    )
    add_time(-1, 'preprocess', (time.time() - start))

    total_method_time = time.time()

    final_results = global_process(
        config, time_limit, end_time,
        method, cluster_method
    )
    add_time(-1, 'total_t_run', (time.time() - start))
    total_method_time = time.time() - total_method_time

    # Print objectives of evaluation part
    (obj_nodes, obj_delta) = mdl.get_obj_value_host()
    it.results_file.write('Number of nodes : %d, Ampli max : %f\n' % (
        obj_nodes, obj_delta))
    it.results_file.write('Number of overloads : %d\n' % final_results['nb_overloads'])
    it.results_file.write('Total execution time : %f s\n\n' % (total_method_time))

    it.clustering_file.write('\nFinal k : %d' % it.my_instance.nb_clusters)

    # Save evaluation results in files
    node_results = node.get_nodes_load_info(
        it.my_instance.df_host_meta)
    it.global_results = pd.DataFrame([{
        'c1': obj_nodes,
        'c2': obj_delta,
        'c3': node_results['load_var'].mean(),
        'c4': final_results['nb_overloads']
    }])

    # Plot nodes consumption
    # TODO plot from it.my_instance.df_host_evo
    # node_evo_fig = plot.plot_containers_groupby_nodes(
    #     it.my_instance.df_indiv,
    #     it.my_instance.df_host_meta[it.metrics[0]].max(),
    #     it.my_instance.sep_time,
    #     title='Initial Node consumption')
    # node_evo_fig.savefig(output_path + '/init_node_plot.svg')

    # Plot clustering & allocation for 1st part
    # plot_before_loop = False
    # if plot_before_loop:
    #     spec_containers = False
    #     if spec_containers:
    #         ctnr.show_specific_containers(working_df_indiv, df_indiv_clust,
    #                                       labels_)
    #     show_clustering = True
    #     if show_clustering:
    #         working_df_indiv = it.my_instance.df_indiv.loc[
    #             it.my_instance.df_indiv[it.tick_field] <= it.my_instance.sep_time
    #         ]
    #         clust_node_fig = plot.plot_clustering_containers_by_node(
    #             working_df_indiv, it.my_instance.dict_id_c,
    #             labels_, filter_big=True)
    #         clust_node_fig.savefig(output_path + '/clust_node_plot.svg')
    #         first_clust_fig = plot.plot_clustering(df_indiv_clust, it.my_instance.dict_id_c,
    #                                                title='Clustering on first half part')
    #         first_clust_fig.savefig(output_path + '/first_clust_plot.svg')

    # Test allocation use case
    # TODO specific window not taken into account
    # if config['allocation']['enable']:
    #     logging.info('Performing allocation ... \n')
    #     print(alloc.check_constraints(
    #         working_df_indiv, config['allocation']))
    # else:
    #     logging.info('We do not perform allocation \n')

    main_time = time.time() - main_time
    print('\nTotal computing time : %f s' % (main_time))
    add_time(-1, 'total_time', main_time)
    node.plot_data_all_nodes(
        it.metrics[0],
        it.my_instance.df_host_meta[it.metrics[0]].max(),
        it.my_instance.sep_time).savefig(
        output_path + '/node_usage_evo.svg')
    it.my_instance.df_host_evo.to_csv(output_path + '/node_usage_evo.csv', index=False)
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

    :param path: initial config file path
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
    :return: tuple with needed parameters and path to output
    :rtype: Tuple[Dict, str]
    """
    # TODO what if we want tick < tau ?
    print('Preprocessing ...')
    (config, output_path, parent_path) = it.read_params(
        path, k, tau, method, cluster_method, param, output, use_kafka)
    config = spec_params(config, [tolclust, tolplace])
    logging.basicConfig(filename=output_path + '/logs.log', filemode='w',
                        format='%(message)s', level=logging.INFO)
    plt.style.use('bmh')
    it.results_file.write('Method used : %s\n' % method)

    # Init containers & nodes data, then Instance
    logging.info('Loading data and creating Instance (Instance information are in results file)\n')
    # reader.consume_all_data(config)
    if it.use_kafka and config['csv']:
        # TODO check this if statement
        reader.consume_all_data(config)
        reader.delete_kafka_topic(config)
        reader.csv_to_stream(parent_path, config)

    reader.init_reader(parent_path)
    it.my_instance = Instance(parent_path, config)
    it.my_instance.print_times()

    return (config, output_path)


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


def global_process(
        config, time_limit, end_time, method, cluster_method
):
    """Perform the global process, from analysis to loops, and retrieve results.

    :return: _description_
    :rtype: Dict
    """
    results = {}
    results['nb_overloads'] = 0
    print('Ready for new data')
    if it.use_kafka and not config['csv']:
        it.my_instance.get_node_information()
        Instance.start_stream()

    # Analysis period
    start = time.time()
    labels_ = analysis_period(config, method)
    add_time(-1, 'total_t_obs', (time.time() - start))
    tmin = it.my_instance.df_indiv[it.tick_field].min()
    tmax = it.my_instance.df_indiv[it.tick_field].max()

    # Prepare the loop
    # it.results_file.write('Loop mode : %s\n' % mode)
    logging.info('Beginning the loop process ...\n')

    (working_df_indiv, df_clust, w, u, v) = build_matrices(
        tmin, tmax, labels_
    )

    # Build initial optimization model in pre loop using analysis data
    start = time.time()
    (clust_model, place_model,
        clustering_dual_values, placement_dual_values) = pre_loop(
        working_df_indiv, df_clust,
        w, u, config['loop']['constraints_dual'], v,
        cluster_method, config['optimization']['solver']
    )
    add_time(0, 'total_loop', (time.time() - start))
    tmax += config['loop']['tick']
    tmin = tmax - (it.my_instance.window_duration - 1)

    run_period(
        tmin, tmax, time_limit, end_time, config, cluster_method, df_clust, labels_,
        clust_model, place_model, clustering_dual_values, placement_dual_values
    )

    return results


def analysis_period(config, method):
    """Perform all needed process during analysis period (T_init).

    :param config: parameters set
    :type config: Dict
    :param method: method used to solve initial problem
    :type method: str
    :return: Clustering labels
    :rtype: List
    """
    current_data = reader.get_next_data(
        0, it.my_instance.sep_time, it.my_instance.sep_time + 1
    )
    if it.use_kafka and current_data.empty and it.end:
        sys.exit(0)

    it.my_instance.df_indiv = current_data
    it.my_instance.df_host = current_data.groupby(
        [current_data[it.tick_field], current_data[it.host_field]],
        as_index=False).agg(it.dict_agg_metrics)

    if not it.use_kafka:
        it.my_instance.set_host_meta(config['host_meta_path'])
    it.my_instance.set_meta_info()
    it.my_instance.init_host_evo()
    df_indiv_clust = pd.DataFrame()
    labels_ = []

    if method == 'init':
        return (it.my_instance.df_host,
                df_indiv_clust, labels_)
    elif method in ['heur', 'loop']:

        # Compute starting point
        # (n_iter, start_point, end_point) = compute_analysis_points()

        it.my_instance.working_df_indiv = it.my_instance.df_indiv

        # First clustering part
        logging.info('Starting first clustering ...')
        print('Starting first clustering ...')
        start = time.time()
        (df_indiv_clust, it.my_instance.dict_id_c) = clt.build_matrix_indiv_attr(
            it.my_instance.working_df_indiv)

        labels_ = clt.perform_clustering(
            df_indiv_clust, config['clustering']['algo'], it.my_instance.nb_clusters)
        df_indiv_clust['cluster'] = labels_
        it.my_instance.nb_clusters = labels_.max() + 1

        # TODO improve this part (distance...)
        cluster_profiles = clt.get_cluster_mean_profile(
            df_indiv_clust)
        cluster_vars = clt.get_cluster_variance(cluster_profiles)

        cluster_var_matrix = clt.get_sum_cluster_variance(
            cluster_profiles, cluster_vars)
        it.results_file.write('\nClustering computing time : %f s\n\n' %
                              (time.time() - start))
        add_time(-1, 'clustering_0', (time.time() - start))

        # First placement part
        if config['placement']['enable'] and config['analysis']['placement_recompute']:
            logging.info('Performing placement ... \n')
            start = time.time()
            place.allocation_distant_pairwise(
                cluster_var_matrix, labels_)
            it.results_file.write(
                '\nPlacement heuristic time : %f s\n\n' % (time.time() - start)
            )
            add_time(-1, 'placement_0', (time.time() - start))
        else:
            print('We keep the initial placement solution \n')
            logging.info('We do not perform placement \n')
    elif method == 'spread':
        place.allocation_spread(it.my_instance.nb_nodes)
    elif method == 'iter-consol':
        place.allocation_spread()
    it.my_instance.df_host_evo = it.my_instance.df_host.loc[
        it.my_instance.df_host[it.tick_field] <= it.my_instance.sep_time
    ]
    print(it.my_instance.df_indiv)
    return labels_


def compute_analysis_points():
    """Compute number of iterations, starting and ending point for analyis."""
    n_iter = math.floor((
        it.my_instance.sep_time + 1 - it.my_instance.df_host[it.tick_field].min()
    ) / it.my_instance.window_duration)
    start_point = (
        it.my_instance.sep_time - n_iter * it.my_instance.window_duration
    ) + 1
    end_point = (
        start_point + it.my_instance.window_duration - 1
    )
    it.my_instance.working_df_indiv = it.my_instance.df_indiv[
        (it.my_instance.
            df_indiv[it.tick_field] >= start_point) & (
            it.my_instance.df_indiv[it.tick_field] <= end_point)
    ]


def run_period(
    tmin, tmax, time_limit, end_time, config, cluster_method, df_clust, labels_,
    clust_model, place_model, clustering_dual_values, placement_dual_values
):
    """Perform loop process, getting new data and evaluating (updating) solutions.

    :param tmin: _description_
    :type tmin: _type_
    :param tmax: _description_
    :type tmax: _type_
    :param time_limit: _description_
    :type time_limit: _type_
    :param end_time: _description_
    :type end_time: _type_
    :param config: _description_
    :type config: _type_
    :param cluster_method: _description_
    :type cluster_method: _type_
    :param df_clust: _description_
    :type df_clust: _type_
    :param labels_: _description_
    :type labels_: _type_
    :param clust_model: _description_
    :type clust_model: _type_
    :param place_model: _description_
    :type place_model: _type_
    :param clustering_dual_values: _description_
    :type clustering_dual_values: _type_
    :param placement_dual_values: _description_
    :type placement_dual_values: _type_
    """
    fig_node, ax_node = plot.init_nodes_plot(
        it.my_instance.df_indiv, it.my_instance.dict_id_n, it.my_instance.sep_time,
        it.my_instance.df_host_meta[it.metrics[0]].max()
    )
    fig_clust, ax_clust = plot.init_plot_clustering(
        df_clust, it.my_instance.dict_id_c)

    cluster_profiles = clt.get_cluster_mean_profile(df_clust)
    fig_mean_clust, ax_mean_clust = plot.init_plot_cluster_profiles(
        cluster_profiles
    )

    total_nb_overloads = 0
    total_loop_time = 0.0
    loop_nb = 1
    nb_clust_changes = 0
    nb_place_changes = 0
    current_time = it.my_instance.sep_time

    try:
        while it.s_entry:
            if time_limit is not None and time.time() >= end_time:
                it.end = True
                it.s_entry = False
                break

            loop_time = time.time()
            logging.info('\n # Enter loop number %d #\n' % loop_nb)
            it.results_file.write('\n # Loop number %d #\n' % loop_nb)
            it.optim_file.write('\n # Enter loop number %d #\n' % loop_nb)
            print('\n # Enter loop number %d #\n' % loop_nb)

            previous_timestamp = it.my_instance.df_indiv[it.tick_field].max()
            # TODO Integrate back progress no loop in here
            current_data = reader.get_next_data(
                current_time, config['loop']['tick'],
                config['loop']['tick'] - current_time + 1
            )
            current_time += config['loop']['tick']
            it.my_instance.df_indiv = pd.concat([
                it.my_instance.df_indiv,
                current_data[~current_data[it.tick_field].isin(
                    it.my_instance.df_indiv[it.tick_field].unique())]
            ], ignore_index=True)
            new_df_host = current_data.groupby(
                [current_data[it.tick_field], it.host_field], as_index=False
            ).agg(it.dict_agg_metrics)
            new_df_host = new_df_host.astype({
                it.host_field: str,
                it.tick_field: int}
            )
            existing_machine_ids = it.my_instance.df_host[
                it.my_instance.df_host[it.tick_field] == previous_timestamp
            ][it.host_field].unique()
            missing_machine_ids = set(existing_machine_ids) - set(
                new_df_host[it.host_field])

            missing_rows = pd.DataFrame({
                'timestamp': int(current_time),
                'machine_id': list(missing_machine_ids),
                'cpu': 0.0
            })
            new_df_host = pd.concat([new_df_host, missing_rows], ignore_index=True)
            # new_df_host.set_index([it.tick_field, it.host_field], inplace=True, drop=False)
            it.my_instance.df_host = pd.concat([
                it.my_instance.df_host,
                new_df_host[~new_df_host[it.tick_field].isin(
                    it.my_instance.df_host[it.tick_field].unique())]
            ], ignore_index=True)
            it.my_instance.df_host_evo = pd.concat([
                it.my_instance.df_host_evo,
                new_df_host[~new_df_host[it.tick_field].isin(
                    it.my_instance.df_host_evo[it.tick_field].unique())]
            ], ignore_index=True)
            (working_df_indiv, df_clust, w, u, v) = build_matrices(
                tmin, tmax, labels_
            )
            cluster_profiles = clt.get_cluster_mean_profile(df_clust)
            (init_loop_obj_nodes, init_loop_obj_delta) = mdl.get_obj_value_indivs(
                working_df_indiv)
            nb_clust_changes_loop = 0
            nb_place_changes_loop = 0
            (nb_clust_changes_loop, nb_place_changes_loop,
                init_loop_silhouette, end_loop_silhouette,
                clust_conf_nodes, clust_conf_edges, clust_max_deg, clust_mean_deg,
                place_conf_nodes, place_conf_edges, place_max_deg, place_mean_deg,
                clust_model, place_model,
                clustering_dual_values, placement_dual_values,
                df_clust, cluster_profiles, labels_) = eval_sols(
                working_df_indiv, cluster_method,
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

            # update clustering & node consumption plot
            plot.update_clustering_plot(
                fig_clust, ax_clust, df_clust, it.my_instance.dict_id_c)
            plot.update_cluster_profiles(
                fig_mean_clust, ax_mean_clust, cluster_profiles,
                sorted(working_df_indiv[it.tick_field].unique()))
            plot.update_nodes_plot(
                fig_node, ax_node, working_df_indiv, it.my_instance.dict_id_n)

            working_df_indiv = it.my_instance.df_indiv[
                (it.my_instance.
                    df_indiv[it.tick_field] >= tmin) & (
                    it.my_instance.df_indiv[it.tick_field] <= tmax)]
            working_df_host = working_df_indiv.groupby(
                [working_df_indiv[it.tick_field], it.host_field],
                as_index=False).agg(it.dict_agg_metrics)

            if loop_nb > 1:
                it.my_instance.df_host_evo = pd.concat(
                    [
                        it.my_instance.df_host_evo,
                        working_df_host[~working_df_host[it.tick_field].isin(
                            it.my_instance.df_host_evo[it.tick_field].unique())]
                    ], ignore_index=True
                )

            loop_time = (time.time() - loop_time)
            (end_loop_obj_nodes, end_loop_obj_delta) = mdl.get_obj_value_host()
            it.results_file.write('Loop delta before changes : %f\n' % init_loop_obj_delta)
            it.results_file.write('Loop delta after changes : %f\n' % end_loop_obj_delta)
            it.results_file.write('Loop time : %f s\n' % loop_time)
            add_time(loop_nb, 'total_loop', loop_time)
            total_loop_time += loop_time

            # Save loop indicators in df
            it.loop_results = pd.concat(
                [
                    it.loop_results if not it.loop_results.empty else None,
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
            tmax += config['loop']['tick']
            tmin = tmax - (it.my_instance.window_duration - 1)
            loop_nb += 1
    finally:
        # Close data reader before exit application
        reader.close_reader()
        end_loop(
            working_df_indiv, tmin, nb_clust_changes, nb_place_changes,
            total_nb_overloads, total_loop_time, loop_nb
        )


def signal_handler_sigint(signal_number, frame):
    """Handle for exiting application via signal."""
    Instance.stop_stream()
    print('Exit application')
    it.s_entry = False
    it.end = True


def signal_handler_sigtstp(signal_number, frame):
    """Handle for exiting application via signal."""
    Instance.stop_stream()
    print('Exit application')
    it.s_entry = False
    it.end = True


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
    nb_overload = 0
    nb_clust_changes = 0
    nb_place_changes = 0

    df_indiv = instance.df_indiv[instance.df_indiv[it.tick_field] == int(tick)].copy()
    # df_indiv = instance.df_indiv[instance.df_indiv[it.tick_field] >= tmin].copy()
    df_host_tick = df_indiv.groupby(
        [df_indiv[it.tick_field], it.host_field],
        as_index=False).agg(it.dict_agg_metrics)

    host_overload = node.check_capacities(df_host_tick, instance.df_host_meta)

    it.my_instance.df_host_evo = pd.concat([
        it.my_instance.df_host_evo, df_host_tick
    ], ignore_index=True)

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
    return (nb_overload, loop_nb, nb_clust_changes, nb_place_changes)


def pre_loop(
    working_df_indiv, df_clust, w, u, constraints_dual, v, cluster_method, solver
):
    """Build optimization problems and solve them with T_init solutions.

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
    print('Building first clustering model ...')
    start = time.time()
    clust_model = mdl.Model(
        1,
        working_df_indiv, it.metrics[0],
        it.my_instance.dict_id_c,
        nb_clusters=it.my_instance.nb_clusters,
        w=w, sol_u=u
    )
    add_time(0, 'build_clustering_model', (time.time() - start))
    start = time.time()
    print('Solving first clustering ...')
    logging.info('# Clustering evaluation #')
    logging.info('Solving linear relaxation ...')
    add_time(0, 'solve_clustering_model', (time.time() - start))
    logging.info('Clustering problem not evaluated yet\n')
    clust_model.solve(solver)
    clustering_dual_values = mdl.fill_dual_values(clust_model)

    if cluster_method == 'stream-km':
        it.streamkm_model = Streamkm(
            coresetsize=it.my_instance.nb_containers,
            length=it.my_instance.time * it.my_instance.nb_containers,
            seed=it.my_instance.nb_clusters,
        )
        it.streamkm_model.partial_fit(df_clust.drop('cluster', axis=1))
        _, labels_ = it.streamkm_model.get_final_clusters(
            it.my_instance.nb_clusters, seed=it.my_instance.nb_clusters)
        df_clust['cluster'] = labels_

    # TODO improve this part (distance...)
    cluster_profiles = clt.get_cluster_mean_profile(
        df_clust)
    cluster_vars = clt.get_cluster_variance(cluster_profiles)

    cluster_var_matrix = clt.get_sum_cluster_variance(
        cluster_profiles, cluster_vars)
    dv = ctnr.build_var_delta_matrix_cluster(
        df_clust, cluster_var_matrix, it.my_instance.dict_id_c)

    # evaluate placement
    logging.info('# Placement evaluation #')
    print('Building first placement model ...')
    start = time.time()
    place_model = mdl.Model(
        2,
        working_df_indiv, it.metrics[0],
        it.my_instance.dict_id_c,
        it.my_instance.dict_id_n,
        it.my_instance.df_host_meta,
        dv=dv, sol_u=u, sol_v=v
    )
    add_time(0, 'build_placement_model', (time.time() - start))
    start = time.time()
    print('Solving first placement ...')
    logging.info('Placement problem not evaluated yet\n')
    add_time(0, 'solve_placement_model', (time.time() - start))
    place_model.solve(solver)
    placement_dual_values = mdl.fill_dual_values(place_model)

    return (clust_model, place_model,
            clustering_dual_values, placement_dual_values)


def build_matrices(tmin, tmax, labels_):
    """Build period dataframe and matrices to be used.

    :param tmin: starting time of current window
    :type tmin: int
    :param tmax: final time of current window
    :type tmax: int
    :param labels_: clustering labels
    :type labels_: List
    :return: current loop data, clustering data, dissimilarity matrix and adjacency matrices
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, np.array, np.array, np.array]
    """
    working_df_indiv = it.my_instance.df_indiv[
        (it.my_instance.
         df_indiv[it.tick_field] >= tmin) & (
            it.my_instance.df_indiv[it.tick_field] <= tmax)]
    (df_clust, it.my_instance.dict_id_c) = clt.build_matrix_indiv_attr(
        working_df_indiv)
    w = clt.build_similarity_matrix(df_clust)
    df_clust['cluster'] = labels_
    u = clt.build_adjacency_matrix(labels_)
    v = place.build_placement_adj_matrix(
        working_df_indiv, it.my_instance.dict_id_c)

    return (working_df_indiv, df_clust, w, u, v)


def eval_sols(
        working_df_indiv,
        cluster_method, w, u, v, clust_model, place_model,
        constraints_dual, clustering_dual_values, placement_dual_values,
        tol_clust, tol_move_clust, tol_open_clust, tol_place, tol_move_place,
        df_clust, cluster_profiles, labels_, loop_nb, solver
):
    """Evaluate clustering and placement solutions.

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
            w, u,
            clust_model, clustering_dual_values, constraints_dual,
            tol_clust, tol_move_clust, tol_open_clust,
            df_clust, cluster_profiles, labels_, loop_nb, solver)
    elif cluster_method == 'kmeans-scratch':
        (dv, nb_clust_changes_loop,
            init_loop_silhouette, end_loop_silhouette,
            clustering_dual_values, clust_model,
            clust_conf_nodes, clust_conf_edges,
            clust_max_deg, clust_mean_deg) = loop_kmeans(
                df_clust, labels_
        )
    elif cluster_method == 'stream-km':
        (dv, nb_clust_changes_loop,
            init_loop_silhouette, end_loop_silhouette,
            clustering_dual_values, clust_model,
            clust_conf_nodes, clust_conf_edges,
            clust_max_deg, clust_mean_deg) = stream_km(
                df_clust, labels_
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
        working_df_indiv,
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
    w, u, clust_model, clustering_dual_values, constraints_dual,
    tol_clust, tol_move_clust, tol_open_clust,
    df_clust, cluster_profiles, labels_, loop_nb, solver
):
    """Evaluate current clustering solution and update it if needed.

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
    print('# Clustering evaluation #')

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
    cluster_profiles = clt.get_cluster_mean_profile(
        df_clust)

    logging.info('Checking for changes in clustering dual values ...')
    start = time.time()
    (moving_containers,
     clust_conflict_nodes,
     clust_conflict_edges,
     clust_max_deg, clust_mean_deg) = mdl.get_moving_containers_clust(
        clust_model, clustering_dual_values,
        tol_clust, tol_move_clust,
        it.my_instance.nb_containers, it.my_instance.dict_id_c,
        df_clust, cluster_profiles)
    add_time(loop_nb, 'get_moves_clustering', (time.time() - start))
    if len(moving_containers) > 0:
        logging.info('Changing clustering ...')
        start = time.time()
        ics = 0.0
        (df_clust, labels_, nb_clust_changes_loop) = clt.change_clustering(
            moving_containers, df_clust, labels_,
            it.my_instance.dict_id_c, tol_open_clust * ics)
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
        df_clust, cluster_var_matrix, it.my_instance.dict_id_c)

    end_loop_silhouette = clt.get_silhouette(df_clust, labels_)

    print('# End of clustering evaluation #')
    return (dv, nb_clust_changes_loop,
            init_loop_silhouette, end_loop_silhouette,
            clustering_dual_values, clust_model,
            clust_conflict_nodes, clust_conflict_edges,
            clust_max_deg, clust_mean_deg)


def eval_placement(
    working_df_indiv, w, u, v, dv,
    placement_dual_values, constraints_dual, place_model, tol_place, tol_move_place,
    nb_clust_changes_loop, loop_nb, solver
):
    """Evaluate current clustering solution and update it if needed.

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
    print('# Placement evaluation #')

    start = time.time()
    # TODO update without re-creating from scratch ? Study
    place_model = mdl.Model(
        2,
        working_df_indiv, it.metrics[0],
        it.my_instance.dict_id_c,
        it.my_instance.dict_id_n,
        it.my_instance.df_host_meta,
        dv=dv, sol_u=u, sol_v=v
    )
    add_time(loop_nb, 'update_placement_model', (time.time() - start))
    it.optim_file.write('solve without any change\n')
    start = time.time()
    place_model.solve(solver)
    add_time(loop_nb, 'solve_placement', (time.time() - start))
    # print('Init placement lp solution : ', place_model.relax_mdl.objective_value)
    moving_containers = []
    moves_list = {}
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
            tol_place, tol_move_place, it.my_instance.nb_containers,
            working_df_indiv, it.my_instance.dict_id_c)
        add_time(loop_nb, 'get_moves_placement', (time.time() - start))
        if len(moving_containers) > 0:
            nb_place_changes_loop = len(moving_containers)
            start = time.time()
            moves_list = place.move_list_containers(
                moving_containers,
                working_df_indiv[it.tick_field].min(),
                working_df_indiv[it.tick_field].max()
            )
            add_time(loop_nb, 'move_placement', (time.time() - start))
            v = place.build_placement_adj_matrix(
                working_df_indiv, it.my_instance.dict_id_c)
            start = time.time()
            place_model.update_adjacency_place_constraints(v)
            place_model.update_obj_place(dv)
            place_model.solve(solver)
            # print('After changes placement lp solution : ', place_model.relax_mdl.objective_value)
            placement_dual_values = mdl.fill_dual_values(place_model)
            add_time(loop_nb, 'solve_new_placement', (time.time() - start))
        else:
            logging.info('No container to move : we do nothing ...\n')

    if it.use_kafka and moves_list:
        move_containers_info(
            moves_list, working_df_indiv[it.tick_field].max()
        )
    print('# End of placement evaluation #')

    return (nb_place_changes_loop, placement_dual_values, place_model,
            place_conf_nodes, place_conf_edges,
            place_max_deg, place_mean_deg)


def move_containers_info(moves_list, current_time):
    """Create the list of containers to move and send it.

    :param moves_list: List of moves to send
    :type moves_list: List
    :param current_time: Current last timestamp to apply moves
    :type moves_list: int
    """
    data = {}
    data['move'] = moves_list
    data[it.tick_field] = str(current_time)
    print('Sending these moving containers :')
    print(data)
    it.kafka_producer.produce(
        it.kafka_topics['docker_replacer'],
        json.dumps(data))
    it.kafka_producer.flush()
    time.sleep(1)


def loop_kmeans(df_clust, labels_):
    """Update clustering via kmeans from scratch.

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
        df_clust.drop('cluster', axis=1), 'kmeans', it.my_instance.nb_clusters)
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
        df_clust, cluster_var_matrix, it.my_instance.dict_id_c)

    end_loop_silhouette = clt.get_silhouette(df_clust, labels_)

    return (dv, nb_clust_changes_loop,
            init_loop_silhouette, end_loop_silhouette,
            {}, None, 0, 0, 0, 0)


def stream_km(df_clust, labels_):
    """Update clustering via kmeans from scratch.

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
        it.my_instance.nb_clusters, seed=it.my_instance.nb_clusters)

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
        df_clust, cluster_var_matrix, it.my_instance.dict_id_c)

    end_loop_silhouette = clt.get_silhouette(df_clust, labels_)

    return (dv, nb_clust_changes_loop,
            init_loop_silhouette, end_loop_silhouette,
            {}, None, 0, 0, 0, 0)


def end_loop(
    working_df_indiv, tmin, nb_clust_changes, nb_place_changes, nb_overload,
    total_loop_time, loop_nb
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
    """
    working_df_host = working_df_indiv.groupby(
        [working_df_indiv[it.tick_field], it.host_field],
        as_index=False).agg(it.dict_agg_metrics)

    (end_loop_obj_nodes, end_loop_obj_delta) = mdl.get_obj_value_host()
    it.results_file.write('Final loop delta : %f\n' % end_loop_obj_delta)

    it.results_file.write('\n### Results of loops ###\n')
    it.results_file.write('Total number of changes in clustering : %d\n' % nb_clust_changes)
    it.results_file.write('Total number of changes in placement : %d\n' % nb_place_changes)
    it.results_file.write('Total number of overload : %d\n' % nb_overload)
    it.results_file.write('Average loop time : %f s\n' % (total_loop_time / loop_nb))

    if loop_nb <= 1:
        it.my_instance.df_host_evo = working_df_indiv.groupby(
            [working_df_indiv[it.tick_field], it.host_field],
            as_index=False).agg(it.dict_agg_metrics)
    else:
        it.my_instance.df_host_evo = pd.concat(
            [
                it.my_instance.df_host_evo,
                working_df_host[~working_df_host[it.tick_field].isin(
                    it.my_instance.df_host_evo[it.tick_field].unique())]
            ], ignore_index=True
        )


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
            it.times_df if not it.times_df.empty else None,
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
