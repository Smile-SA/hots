"""Provide actions specific to containers."""

from itertools import combinations

from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

from . import init as it
from . import plot

# Definition of Container-related functions #


def plot_data_all_containers(df_indiv, metric):
    """Plot a specific metric containers consumption.

    :param df_indiv: Individual consumption data
    :type df_indiv: pd.DataFrame
    :param metric: Metric to display
    :type metric: str
    """
    fig, ax = plt.subplots()
    temp_df = df_indiv.reset_index(drop=True)
    pvt = pd.pivot_table(temp_df, columns=it.indiv_field,
                         index=it.tick_field, aggfunc='sum', values=metric)
    pvt.plot(ax=ax)
    fig.suptitle('%s use on all containers' % metric)
    plt.draw()


# The following uses plotly : comment for the moment
# def plot_all_data_all_containers_px(df_indiv, sep_time, metrics=None):
#     """Plot all metrics containers consumption.

#     :param df_indiv: _description_
#     :type df_indiv: pd.DataFrame
#     :param sep_time: _description_
#     :type sep_time: int
#     :param metrics: _description_, defaults to None
#     :type metrics: List[str], optional
#     """
#     # TODO several metrics ?
#     # TODO line_shape in params
#     metrics = metrics or it.metrics
#     fig = px.line(df_indiv, x=it.tick_field, y=metrics[0],
#                   color=it.host_field,
#                   line_group=it.indiv_field, hover_name=it.indiv_field,
#                   line_shape='spline', render_mode='svg',
#                   title='Resource usage on all individuals')
#     fig.add_vline(sep_time, line={'color': 'red', 'dash': 'dashdot'})
#     fig.show(it.renderer)


def plot_all_data_all_containers(df_indiv, sep_time, metrics=None):
    """Plot all metrics containers consumption.

    :param df_indiv: Individual consumption data
    :type df_indiv: pd.DataFrame
    :param sep_time: Separation time (Analysis / Running)
    :type sep_time: int
    :param metrics: Metrics to display, defaults to None
    :type metrics: List[str], optional
    :return: Figure object
    :rtype: plt.Figure
    """
    # TODO several metrics ?
    plt.style.use('bmh')
    metrics = metrics or it.metrics
    fig = plt.figure()
    fig.suptitle('Resource usage on all containers')
    gs = gridspec.GridSpec(len(metrics), 1)
    ax_ = []
    ai = 0
    for metric in metrics:
        ax_.append(fig.add_subplot(gs[ai, 0]))
        ax_[ai].set_title(metric)
        ax_[ai].set(xlabel='time (s)', ylabel=metric)
        # ax_[ai].grid()
        temp_df = df_indiv.reset_index(drop=True)
        pvt = pd.pivot_table(temp_df, columns=it.indiv_field,
                             index=it.tick_field, aggfunc='sum', values=metric)
        pvt.plot(ax=ax_[ai], legend=False)
        # pvt.plot(ax=ax_[ai])
        # ax_[ai].axvline(x=sep_time, color='red', linestyle='--')
        ai += 1

    fig.align_labels()
    plt.draw()
    return fig


def build_var_delta_matrix(df_indiv, dict_id_c):
    """Build variance of deltas matrix.

    :param df_indiv: Individual consumption data
    :type df_indiv: pd.DataFrame
    :param dict_id_c: Mapping dict container ID / numerical ID
    :type dict_id_c: Dict
    :return: Result matrix
    :rtype: np.array
    """
    c = df_indiv[it.indiv_field].nunique()
    vars_matrix = np.zeros((c, c), dtype=float)

    for (c1_s, c2_s) in combinations(df_indiv[it.indiv_field].unique(), 2):
        vals_c1 = df_indiv.loc[df_indiv[it.indiv_field] == c1_s,
                               it.metrics[0]].values
        vals_c2 = df_indiv.loc[df_indiv[it.indiv_field] == c2_s,
                               it.metrics[0]].values
        c1 = [k for k, v in dict_id_c.items() if v == c1_s][0]
        c2 = [k for k, v in dict_id_c.items() if v == c2_s][0]
        vars_matrix[c1, c2] = (vals_c1 + vals_c2).var()
        vars_matrix[c2, c1] = vars_matrix[c1, c2]
    return vars_matrix


def build_var_delta_matrix_cluster(df_clust, cluster_var_matrix, dict_id_c):
    """Build variance of deltas matrix from cluster.

    :param df_clust: Formated consumption data
    :type df_clust: pd.DataFrame
    :param cluster_var_matrix: Clusters variance
    :type cluster_var_matrix: np.array
    :param dict_id_c: Mapping dict container ID / numerical ID
    :type dict_id_c: Dict
    :return: Individuals variances from cluster variances
    :rtype: np.array
    """
    c = len(df_clust)
    vars_matrix = np.zeros((c, c), dtype=float)

    for (c1_s, c2_s) in combinations(list(df_clust.index.values), 2):
        c1 = [k for k, v in dict_id_c.items() if v == c1_s][0]
        c2 = [k for k, v in dict_id_c.items() if v == c2_s][0]
        vars_matrix[c1, c2] = cluster_var_matrix[
            df_clust.at[c1_s, 'cluster'], df_clust.at[c2_s, 'cluster']]
        vars_matrix[c2, c1] = vars_matrix[c1, c2]
    return vars_matrix


def build_vars_matrix_indivs(df_clust, vars_, dict_id_c):
    """Build containers matrix with clusters variance.

    :param df_clust: Formated consumption data
    :type df_clust: pd.DataFrame
    :param vars_: Clusters variance
    :type vars_: np.array
    :param dict_id_c: Mapping dict container ID / numerical ID
    :type dict_id_c: Dict
    :return: Cluster sum variance matrix
    :rtype: np.array
    """
    c = len(df_clust)
    vars_matrix = np.zeros((c, c), dtype=float)
    for (c1_s, c2_s) in combinations(list(df_clust.index.values), 2):
        c1 = [k for k, v in dict_id_c.items() if v == c1_s][0]
        c2 = [k for k, v in dict_id_c.items() if v == c2_s][0]
        vars_matrix[c1, c2] = vars_[
            df_clust.at[c1_s, 'cluster']] + vars_[
                df_clust.at[c2_s, 'cluster']]
        vars_matrix[c2, c1] = vars_matrix[c1, c2]
    return vars_matrix


def show_specific_containers(
    working_df_indiv, df_indiv_clust, labels_
):
    """Show specific (user input) containers.

    :param working_df_indiv: Individual consumption data
    :type working_df_indiv: pd.DataFrame
    :param df_indiv_clust: Formated consumption data
    :type df_indiv_clust: pd.DataFrame
    :param labels_: List of assigned clusters to individuals
    :type labels_: List
    """
    print('Enter list of containers to show separated by a comma :')
    containers_input = input()
    containers_toshow = [int(s) for s in containers_input.split(',')]
    plot.plot_clustering_containers_by_node_spec_cont(
        working_df_indiv, it.my_instance.dict_id_c,
        labels_, containers_toshow
    )
    plot.plot_clustering_spec_cont(df_indiv_clust,
                                   it.my_instance.dict_id_c,
                                   containers_toshow,
                                   title='Clustering on first half part')
