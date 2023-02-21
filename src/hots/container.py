"""
=========
hots container
=========

Provide actions specific to containers (plot containers data,
build dictionnary for container IDs ...)
"""

from itertools import combinations
from typing import Dict, List

from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import plotly.express as px

import init as it
import plot
import instance

# Definition of Container-related functions #


def plot_data_all_containers(df_indiv: pd.DataFrame, metric: str):
    """Plot a specific metric containers consumption."""
    fig, ax = plt.subplots()
    temp_df = df_indiv.reset_index(drop=True)
    pvt = pd.pivot_table(temp_df, columns=it.indiv_field,
                         index=it.tick_field, aggfunc='sum', values=metric)
    pvt.plot(ax=ax)
    fig.suptitle('%s use on all containers' % metric)
    plt.draw()


def plot_all_data_all_containers_px(
        df_indiv: pd.DataFrame,
        sep_time: int, metrics: List[str] = None):
    """Plot all metrics containers consumption."""
    # TODO several metrics ?
    # TODO line_shape in params
    metrics = metrics or it.metrics
    fig = px.line(df_indiv, x=it.tick_field, y=metrics[0],
                  color=it.host_field,
                  line_group=it.indiv_field, hover_name=it.indiv_field,
                  line_shape='spline', render_mode='svg',
                  title='Resource usage on all individuals')
    fig.add_vline(sep_time, line={'color': 'red', 'dash': 'dashdot'})
    fig.show(it.renderer)


def plot_all_data_all_containers(
        df_indiv: pd.DataFrame,
        sep_time: int, metrics: List[str] = None) -> plt.Figure:
    """Plot all metrics containers consumption."""
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


def build_dict_id_containers(df_indiv: pd.DataFrame) -> Dict:
    """Build dictionnary for corresponding IDs and indexes."""
    dict_id_c = {}
    i = 0
    for key in df_indiv.container_id.unique():
        dict_id_c[i] = key
        i += 1

    return dict_id_c


def build_var_delta_matrix(df_indiv: pd.DataFrame, dict_id_c) -> np.array:
    """Build variance of deltas matrix."""
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


def build_var_delta_matrix_cluster(
        df_clust: pd.DataFrame, cluster_var_matrix: np.array, dict_id_c: Dict) -> np.array:
    """Build variance of deltas matrix from cluster."""
    c = len(df_clust)
    vars_matrix = np.zeros((c, c), dtype=float)

    for (c1_s, c2_s) in combinations(list(df_clust.index.values), 2):
        c1 = [k for k, v in dict_id_c.items() if v == c1_s][0]
        c2 = [k for k, v in dict_id_c.items() if v == c2_s][0]
        vars_matrix[c1, c2] = cluster_var_matrix[
            df_clust.at[c1_s, 'cluster'], df_clust.at[c2_s, 'cluster']]
        vars_matrix[c2, c1] = vars_matrix[c1, c2]
    return vars_matrix


def build_vars_matrix_indivs(
        df_clust: pd.DataFrame, vars_: np.array, dict_id_c: Dict) -> np.array:
    """Build containers matrix with clusters variance."""
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


def show_specific_containers(working_df_indiv: pd.DataFrame,
                             df_indiv_clust: pd.DataFrame,
                             my_instance: instance.Instance, labels_: List):
    """Show specific (user input) containers."""
    print('Enter list of containers to show separated by a comma :')
    containers_input = input()
    containers_toshow = [int(s) for s in containers_input.split(',')]
    plot.plot_clustering_containers_by_node_spec_cont(
        working_df_indiv, my_instance.dict_id_c,
        labels_, containers_toshow
    )
    plot.plot_clustering_spec_cont(df_indiv_clust,
                                   my_instance.dict_id_c,
                                   containers_toshow,
                                   title='Clustering on first half part')
