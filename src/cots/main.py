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
from . import node
from . import placement as place
from . import plot
from .instance import Instance


# TODO add 'help' message
@click.command()
@click.option('--path', required=True, type=click.Path(exists=True))
@click.option('--k', required=False, type=int)
@click.option('--tau', required=False, type=int)
def main(path, k, tau):
    """Perform all things of methodology."""
    # Initialization part
    main_time = time.time()

    if not path[-1] == '/':
        path += '/'
    (config, output_path) = it.read_params(path, k, tau)
    logging.basicConfig(filename=output_path + '/logs.log', filemode='w',
                        format='%(message)s', level=logging.INFO)
    plt.style.use('bmh')

    # Init containers & nodes data, then Instance
    logging.info('Loading data and creating Instance (Instance information are in results file)\n')
    my_instance = Instance(path, config)

    # Use pyomo model => to be fully applied after tests
    # my_model = model.create_model(config['optimization']['model'], my_instance)
    # model.solve_model(my_model, 'glpk')

    # Plot initial data (containers & nodes consumption)
    ctnr.plot_all_data_all_containers(
        my_instance.df_indiv, sep_time=my_instance.sep_time)

    node_evo_fig = plot.plot_containers_groupby_nodes(
        my_instance.df_indiv,
        my_instance.df_host_meta[it.metrics[0]].max(),
        my_instance.sep_time,
        title='Initial Node consumption')
    node_evo_fig.savefig(output_path + '/init_node_plot.svg')

    # Print real objective value of second part if no loop
    it.results_file.write(
        'Real objective value of second part with initial dataset placement :\n')
    (init_obj_nodes, init_obj_delta) = mc.get_obj_value_heuristic(
        my_instance.df_indiv,
        my_instance.eval_time,
        my_instance.df_indiv[it.tick_field].max())
    it.results_file.write('Number of nodes : %d, Delta : %f\n\n' % (
        init_obj_nodes, init_obj_delta))
    it.main_results.append(init_obj_nodes)
    it.main_results.append(init_obj_delta)
    node_results = get_node_results(my_instance, 'init_')

    # Print real objective value of second part with spread technique
    spread_time = time.time()
    place.allocation_spread(my_instance, my_instance.nb_nodes)
    # TODO progress in eval window for check capacities
    # spread_df_host = progress_time_noloop(my_instance)
    # print(spread_df_host)
    spread_time = time.time() - spread_time
    it.results_file.write(
        'Real objective value of second part with spread technique :\n')
    (spread_obj_nodes, spread_obj_delta) = mc.get_obj_value_heuristic(
        my_instance.df_indiv,
        my_instance.eval_time,
        my_instance.df_indiv[it.tick_field].max())
    it.results_file.write('Number of nodes : %d, Delta : %f\n' % (
        spread_obj_nodes, spread_obj_delta))
    it.results_file.write('Spread technique time : %f s\n\n' % (spread_time))
    it.main_results.append(spread_obj_nodes)
    it.main_results.append(spread_obj_delta)
    it.main_results.append(spread_time)
    node_results = pd.concat([
        node_results,
        get_node_results(my_instance, 'spread_')],
        axis=1
    )

    # Plot spread result
    node_evo_fig = plot.plot_containers_groupby_nodes(
        my_instance.df_indiv,
        my_instance.df_host_meta[it.metrics[0]].max(),
        my_instance.sep_time,
        title='Node consumption after spread technique')
    node_evo_fig.savefig(output_path + '/node_spread_plot.svg')

    # Print real objective value of second part with iterative consolidation technique
    pack_spread_time = time.time()
    place.allocation_spread(my_instance)
    # TODO progress in eval window for check capacities
    # spread_df_host = progress_time_noloop(my_instance)
    # print(spread_df_host)
    pack_spread_time = time.time() - pack_spread_time
    it.results_file.write(
        'Real objective value of second part with iterative consolidation technique :\n')
    (pack_spread_obj_nodes, pack_spread_obj_delta) = mc.get_obj_value_heuristic(
        my_instance.df_indiv,
        my_instance.eval_time,
        my_instance.df_indiv[it.tick_field].max())
    it.results_file.write('Number of nodes : %d, Delta : %f\n' % (
        pack_spread_obj_nodes, pack_spread_obj_delta))
    it.results_file.write('Iterative consolidation technique time : %f s\n\n' % (pack_spread_time))
    it.main_results.append(pack_spread_obj_nodes)
    it.main_results.append(pack_spread_obj_delta)
    it.main_results.append(pack_spread_time)
    node_results = pd.concat([
        node_results,
        get_node_results(my_instance, 'iter-consol_')],
        axis=1
    )

    # Plot iterative consolidation result
    node_evo_fig = plot.plot_containers_groupby_nodes(
        my_instance.df_indiv,
        my_instance.df_host_meta[it.metrics[0]].max(),
        my_instance.sep_time,
        title='Node consumption after iterative consolidation technique')
    node_evo_fig.savefig(output_path + '/node_iter-consol_plot.svg')

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

    it.results_file.write('\nClustering computing time : %f s\n\n' %
                          (time.time() - clustering_time))

    # Test allocation use case
    # TODO specific window not taken into account
    if config['allocation']['enable']:
        logging.info('Performing allocation ... \n')
        print(alloc.check_constraints(
            my_instance, working_df_indiv, config['allocation']))
    else:
        logging.info('We do not perform allocation \n')

    # Placement
    containers_grouped = []
    if config['placement']['enable']:
        logging.info('Performing placement ... \n')
        heur_time = time.time()
        containers_grouped = place.allocation_distant_pairwise(
            my_instance, cluster_var_matrix, labels_)
        heur_time = time.time() - heur_time
    else:
        logging.info('We do not perform placement \n')

    # Print real objective value of second part if no loop
    it.results_file.write(
        'Real objective value of second part with heuristic only (without loop)\n')
    (heur_obj_nodes, heur_obj_delta) = mc.get_obj_value_heuristic(
        my_instance.df_indiv,
        my_instance.eval_time,
        my_instance.df_indiv[it.tick_field].max())
    it.results_file.write('Number of nodes : %d, Delta : %f\n' % (
        heur_obj_nodes, heur_obj_delta))
    it.results_file.write('Heuristic time : %f s\n\n' % (heur_time))
    it.main_results.append(heur_obj_nodes)
    it.main_results.append(heur_obj_delta)
    it.main_results.append(heur_time)
    node_results = pd.concat([
        node_results,
        get_node_results(my_instance, 'heur_')],
        axis=1
    )

    # Plot clustering & allocation for 1st part
    plot_before_loop = True
    if plot_before_loop:
        spec_containers = False
        if spec_containers:
            ctnr.show_specific_containers(working_df_indiv, df_indiv_clust,
                                          my_instance, labels_)
        show_clustering = True
        if show_clustering:
            working_df_indiv = my_instance.df_indiv.loc[
                my_instance.df_indiv[it.tick_field] <= my_instance.sep_time
            ]
            clust_node_fig = plot.plot_clustering_containers_by_node(
                working_df_indiv, my_instance.dict_id_c,
                labels_, filter_big=True)
            clust_node_fig.savefig(output_path + '/clust_node_plot.svg')
            first_clust_fig = plot.plot_clustering(df_indiv_clust, my_instance.dict_id_c,
                                                   title='Clustering on first half part')
            first_clust_fig.savefig(output_path + '/first_clust_plot.svg')
        # plot.plot_containers_groupby_nodes(
        #     my_instance.df_indiv,
        #     my_instance.df_host_meta.cpu.max(),
        #     my_instance.sep_time)

    # Plot heuristic result without loop
    node_evo_fig = plot.plot_containers_groupby_nodes(
        my_instance.df_indiv,
        my_instance.df_host_meta[it.metrics[0]].max(),
        my_instance.sep_time,
        title='Node consumption after heuristic without loop')
    node_evo_fig.savefig(output_path + '/node_heur_plot.svg')

    # input('\nEnd of first part, press enter to enter loop ...\n')

    # Test allocation use case
    # TODO specific window not taken into account
    # if config['allocation']['enable']:
    #     logging.info('Performing allocation ... \n')
    #     print(alloc.check_constraints(
    #         my_instance, working_df_indiv, config['allocation']))
    # else:
    #     logging.info('We do not perform allocation \n')

    # Plot heuristic result without loop
    # ctnr.plot_all_data_all_containers(
    #     my_instance.df_indiv, sep_time=my_instance.sep_time)
    # plot.plot_containers_groupby_nodes(
    #     my_instance.df_indiv,
    #     my_instance.df_host_meta[it.metrics[0]].max(),
    #     my_instance.sep_time,
    #     title='Node consumption after heuristic and change allocation')

    # loop 'streaming' progress
    it.results_file.write('\n### Loop process ###\n')
    (fig_node, fig_clust, fig_mean_clust,
     loop_main_results, df_host_evo) = streaming_eval(
        my_instance, df_indiv_clust, labels_,
        containers_grouped, config['loop']['tick'],
        config['loop']['constraints_dual'],
        config['loop']['tol_dual_clust'],
        config['loop']['tol_move_clust'],
        config['loop']['tol_dual_place'],
        config['loop']['tol_move_place'],
        config['loop']['tol_step'])

    fig_node.savefig(output_path + '/node_evo_plot.svg')
    fig_clust.savefig(output_path + '/clust_evo_plot.svg')
    fig_mean_clust.savefig(output_path + '/mean_clust_evo_plot.svg')

    # Plot after the loop
    plot.plot_containers_groupby_nodes(
        my_instance.df_indiv,
        my_instance.df_host_meta[it.metrics[0]].max(),
        my_instance.sep_time,
        title='Node consumption with loop')

    # Print real objective value
    it.results_file.write('Real objective value of second part with loop :\n')
    # (loop_obj_nodes, loop_obj_delta) = mc.get_obj_value_heuristic(
    #     my_instance.df_indiv,
    #     my_instance.eval_time,
    #     my_instance.df_indiv[it.tick_field].max())
    (loop_obj_nodes, loop_obj_delta) = mc.get_obj_value_host(
        df_host_evo)
    it.results_file.write('Number of nodes : %d, Delta : %f\n' % (
        loop_obj_nodes, loop_obj_delta))
    it.main_results.append(loop_obj_nodes)
    it.main_results.append(loop_obj_delta)
    it.main_results.extend(loop_main_results)
    node_results = pd.concat([
        node_results,
        get_node_results(my_instance, 'loop_')],
        axis=1
    )

    main_time = time.time() - main_time
    it.main_results.append(main_time)
    node_results.to_csv(output_path + '/node_results.csv')
    it.loop_results.to_csv(output_path + '/loop_results.csv', index=False)
    write_main_results()
    it.results_file.write('\nTotal computing time : %f s' % (main_time))
    close_files()


# TODO last ticks of data may not be plotted
def streaming_eval(my_instance: Instance, df_indiv_clust: pd.DataFrame,
                   labels_: List, containers_grouped: List, tick: int,
                   constraints_dual: List,
                   tol_clust: float, tol_move_clust: float,
                   tol_place: float, tol_move_place: float, tol_step: float
                   ) -> (plt.Figure, plt.Figure, plt.Figure, List, pd.DataFrame):
    """Define the streaming process for evaluation."""
    fig_node, ax_node = plot.init_nodes_plot(
        my_instance.df_indiv, my_instance.dict_id_n, my_instance.sep_time,
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

    df_host_evo = pd.DataFrame(columns=my_instance.df_host.columns)

    tmin = my_instance.df_indiv[it.tick_field].min()
    tmax = my_instance.sep_time
    current_obj_func = 1
    clustering_dual_values = {}
    placement_dual_values = {}

    total_loop_time = 0.0
    loop_nb = 1
    nb_clust_changes = 0
    nb_place_changes = 0
    end = False

    logging.info('Beginning the loop process ...\n')
    while not end:
        loop_time = time.time()
        logging.info('\n # Enter loop number %d #\n' % loop_nb)
        it.results_file.write('\n # Loop number %d #\n' % loop_nb)
        it.optim_file.write('\n # Enter loop number %d #\n' % loop_nb)
        print('\n # Enter loop number %d #\n' % loop_nb)

        # TODO not fully tested (replace containers)
        # if loop_nb > 1:
        #     progress_time_noloop(my_instance, tmin, tmax)

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
        moving_containers = []

        if loop_nb == 1:
            logging.info('Evaluation of problems with initial solutions')
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
            logging.info('Clustering problem not evaluated yet\n')
            clustering_dual_values = mc.fill_constraints_dual_values(
                cplex_model.relax_mdl, constraints_dual
            )

            # TODO improve this part (use cluster profiles)
            dv = ctnr.build_var_delta_matrix(
                working_df_indiv, my_instance.dict_id_c)

            # evaluate placement
            logging.info('# Placement evaluation #')
            cplex_model = mc.CPXInstance(working_df_indiv,
                                         my_instance.df_host_meta,
                                         my_instance.nb_clusters,
                                         my_instance.dict_id_c,
                                         my_instance.dict_id_n,
                                         obj_func=current_obj_func,
                                         w=w, u=u, v=v, dv=dv, pb_number=3)
            print('Solving linear relaxation ...')
            cplex_model.solve(cplex_model.relax_mdl)
            logging.info('Placement problem not evaluated yet\n')
            placement_dual_values = mc.fill_constraints_dual_values(
                cplex_model.relax_mdl, constraints_dual
            )

        else:  # We have already an evaluated solution

            nb_clust_changes_loop = 0
            nb_place_changes_loop = 0

            # TODO not very practical
            # plot.plot_clustering_containers_by_node(
            #     working_df_indiv, my_instance.dict_id_c, labels_)

            # evaluate clustering solution from optim model
            # mc.eval_clustering(
            #     working_df_indiv,
            #     my_instance,
            #     current_obj_func,
            #     w, u,
            #     clustering_dual_values,
            #     moving_containers
            # )
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

            logging.info('Checking for changes in clustering dual values ...')
            time_get_clust_move = time.time()
            moving_containers = mc.get_moving_containers_clust(
                cplex_model.relax_mdl, clustering_dual_values,
                tol_clust, tol_move_clust,
                my_instance.nb_containers, my_instance.dict_id_c,
                df_clust, cluster_profiles)
            print('Time get changing clustering : ', (time.time() - time_get_clust_move))
            if len(moving_containers) > 0:
                logging.info('Changing clustering ...')
                time_change_clust = time.time()
                (df_clust, labels_, nb_clust_changes_loop) = clt.change_clustering(
                    moving_containers, df_clust, labels_,
                    cluster_profiles, my_instance.dict_id_c)
                u = clt.build_adjacency_matrix(labels_)
                cplex_model = mc.CPXInstance(working_df_indiv,
                                             my_instance.df_host_meta,
                                             my_instance.nb_clusters,
                                             my_instance.dict_id_c,
                                             my_instance.dict_id_n,
                                             obj_func=current_obj_func,
                                             w=w, u=u, pb_number=2)
                print('Time changing clustering : ', (time.time() - time_change_clust))
                logging.info('Solving linear relaxation after changes ...')
                cplex_model.solve(cplex_model.relax_mdl)
                clustering_dual_values = mc.fill_constraints_dual_values(
                    cplex_model.relax_mdl, constraints_dual
                )
            else:
                logging.info('Clustering seems still right ...')
                it.results_file.write('Clustering seems still right ...')

            # Compute new clusters profiles
            cluster_profiles = clt.get_cluster_mean_profile(
                my_instance.nb_clusters,
                df_clust,
                my_instance.window_duration, tmin=tmin)

            # TODO improve this part (use cluster profiles)
            dv = ctnr.build_var_delta_matrix(
                working_df_indiv, my_instance.dict_id_c)

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
            logging.info('# Placement evaluation #')
            cplex_model = mc.CPXInstance(working_df_indiv,
                                         my_instance.df_host_meta,
                                         my_instance.nb_clusters,
                                         my_instance.dict_id_c,
                                         my_instance.dict_id_n,
                                         obj_func=current_obj_func,
                                         w=w, u=u, v=v, dv=dv, pb_number=3)
            print('Solving linear relaxation ...')
            cplex_model.solve(cplex_model.relax_mdl)
            moving_containers = []
            if nb_clust_changes_loop > 0:
                logging.info('Checking for changes in placement dual values ...')
                time_get_move = time.time()
                moving_containers = mc.get_moving_containers(
                    cplex_model.relax_mdl, placement_dual_values,
                    tol_place, tol_move_place, my_instance.nb_containers,
                    working_df_indiv, my_instance.dict_id_c)
                print('Time get moving containers : ', (time.time() - time_get_move))

            if len(moving_containers) > 0:
                nb_place_changes_loop = len(moving_containers)
                time_move_place = time.time()
                place.move_list_containers(moving_containers, my_instance,
                                           working_df_indiv[it.tick_field].min(),
                                           working_df_indiv[it.tick_field].max())
                print('Time moving containers : ', (time.time() - time_move_place))
                cplex_model = mc.CPXInstance(working_df_indiv,
                                             my_instance.df_host_meta,
                                             my_instance.nb_clusters,
                                             my_instance.dict_id_c,
                                             my_instance.dict_id_n,
                                             obj_func=current_obj_func,
                                             w=w, u=u, v=v, dv=dv, pb_number=3)
                print('Solving linear relaxation after changes ...')
                cplex_model.solve(cplex_model.relax_mdl)
                placement_dual_values = mc.fill_constraints_dual_values(
                    cplex_model.relax_mdl, constraints_dual
                )

            else:
                logging.info('No container to move : we do nothing ...\n')

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

            (init_loop_obj_nodes, init_loop_obj_delta) = mc.get_obj_value_heuristic(
                working_df_indiv)
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
            it.results_file.write('Loop time : %f s\n' % (time.time() - loop_time))
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

    df_host_evo = end_loop(my_instance, tmin, nb_clust_changes, nb_place_changes,
                           total_loop_time, loop_nb, df_host_evo)

    return (fig_node, fig_clust, fig_mean_clust,
            [nb_clust_changes, nb_place_changes, total_loop_time, total_loop_time / loop_nb],
            df_host_evo)


def progress_time_noloop(instance: Instance,
                         tmin: int, tmax: int) -> pd.DataFrame:
    """We progress in time without performing the loop, checking node capacities."""
    for tick in range(tmin, tmax + 1):
        df_host_tick = instance.df_indiv.loc[
            instance.df_indiv[it.tick_field] == tick
        ].groupby(
            [instance.df_indiv[it.tick_field], it.host_field],
            as_index=False).agg(it.dict_agg_metrics)
        host_overload = node.check_capacities(df_host_tick, instance.df_host_meta)
        if len(host_overload) > 0:
            print('Overload : We must move containers')
            place.free_full_nodes(instance, host_overload, tick)


def end_loop(my_instance: Instance, tmin: int, nb_clust_changes: int, nb_place_changes: int,
             total_loop_time: float, loop_nb: int, df_host_evo: pd.DataFrame):
    """Perform all stuffs after last loop."""
    working_df_indiv = my_instance.df_indiv[
        (my_instance.
         df_indiv[it.tick_field] >= tmin)]
    working_df_host = working_df_indiv.groupby(
        [working_df_indiv[it.tick_field], it.host_field],
        as_index=False).agg(it.dict_agg_metrics)

    (end_loop_obj_nodes, end_loop_obj_delta) = mc.get_obj_value_host(
        working_df_host)
    it.results_file.write('Final loop delta : %f\n' % end_loop_obj_delta)

    it.results_file.write('\n### Results of loops ###\n')
    it.results_file.write('Total number of changes in clustering : %d\n' % nb_clust_changes)
    it.results_file.write('Total number of changes in placement : %d\n' % nb_place_changes)
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


def get_node_results(instance: Instance, algo: str) -> pd.DataFrame:
    """Compute all wanted additional indicators."""
    eval_df_host = instance.df_host.loc[
        instance.df_host[it.tick_field] >= instance.eval_time
    ]

    result_df = node.get_nodes_load_info(
        eval_df_host, instance.df_host_meta)
    result_df = result_df.add_prefix(algo)

    return result_df


def close_files():
    """Write the final files and close all open files."""
    it.results_file.close()
    it.main_results_file.close()


if __name__ == '__main__':
    main()
