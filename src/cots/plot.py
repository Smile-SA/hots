"""
=========
cots plot
=========

Provide all plotting methods : different clustering results views,
containers data, nodes data, continous plot in evaluation step.
"""

import math
from typing import Dict, List

from matplotlib import gridspec as gridspec
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt

import networkx as nx

import numpy as np

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import scipy.cluster.hierarchy as hac

from . import init as it
from .instance import Instance

# Global variables
# TODO manage big number of colors
colors = ['blue', 'orange', 'green', 'red', 'purple',
          'brown', 'pink', 'gray', 'olive', 'cyan', 'turquoise',
          'chocolate', 'navy', 'lightcoral', 'violet']

other_colors = ['violet', 'lightcoral', 'navy', 'chocolate', 'turquoise']


# TODO better plots (especially "live" plots)
# Functions definitions #


def plot_clustering(df_clust: pd.DataFrame, dict_id_c: Dict,
                    metric: str = None, title: str = None):
    """Plot metric containers consumption, grouped by cluster."""
    metric = metric or it.metrics[0]
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(metric + ' consumption of containers grouped by cluster')
    gs = gridspec.GridSpec(df_clust.cluster.max() + 1, 1)
    ax_ = []
    max_cons = df_clust.drop(labels='cluster', axis=1).values.max()
    for k, data in df_clust.groupby(['cluster']):
        ax_.append(fig.add_subplot(gs[k, 0]))
        ax_[k].set_title('Cluster n°%d' % k, pad=k)
        ax_[k].set(xlabel='time (s)', ylabel=metric)
        ax_[k].grid()
        ax_[k].set_ylim([0, math.ceil(max_cons)])
        for row in data.drop(labels='cluster', axis=1).iterrows():
            c_int = [key for key, v in dict_id_c.items() if v == row[0]][0]
            ax_[k].plot(row[1], colors[k], label=c_int)
        # ax_[k].legend()
        k_patch = mpatches.Patch(color=colors[k], label=len(data))
        ax_[k].legend(handles=[k_patch], loc='upper left')
    plt.draw()
    return fig


def plot_clustering_spec_cont(df_clust: pd.DataFrame, dict_id_c: Dict,
                              containers_toshow: List,
                              metric: str = 'cpu', title: str = None):
    """Plot metric containers consumption, grouped by cluster."""
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(metric + ' consumption of containers grouped by cluster')
    gs = gridspec.GridSpec(df_clust.cluster.max() + 1, 1)
    ax_ = []
    max_cons = df_clust.drop(labels='cluster', axis=1).values.max()
    for k, data in df_clust.groupby(['cluster']):
        ax_.append(fig.add_subplot(gs[k, 0]))
        ax_[k].set_title('Cluster n°%d' % k, pad=k)
        ax_[k].set(xlabel='time (s)', ylabel=metric)
        ax_[k].grid()
        ax_[k].set_ylim([0, math.ceil(max_cons)])
        for row in data.drop(labels='cluster', axis=1).iterrows():
            c_int = [key for key, v in dict_id_c.items() if v == row[0]][0]
            if c_int in containers_toshow:
                ax_[k].plot(row[1], other_colors[k], label=c_int)
            else:
                ax_[k].plot(row[1], colors[k])
        ax_[k].legend()
    plt.draw()


def plot_containers_clustering_together(df_clust: pd.DataFrame,
                                        metric: str = 'cpu'):
    """Plot all containers consumption with their cluster color."""
    fig, ax = plt.subplots()
    fig.suptitle('Containers clustering (' + metric + ')')
    for row in df_clust.iterrows():
        ax.plot(row[1].drop(labels='cluster'),
                color=colors[int(row[1]['cluster'])])
    # ax.legend()
    plt.draw()


def plot_clustering_containers_by_node(
        df_indiv: pd.DataFrame, dict_id_c: Dict, labels_: List,
        filter_big: bool = False, metric: str = None):
    """
    Plot containers consumption grouped by node, one container added above
    another, with their cluster color.
    """
    metric = metric or it.metrics[0]
    if filter_big:
        to_filter = np.bincount(labels_).argmax()
    else:
        to_filter = None
    fig_node_usage = plt.figure()
    fig_node_usage.suptitle(
        metric + ' consumption in each node, containers clustering')
    gs_node_usage = gridspec.GridSpec(math.ceil(
        df_indiv[it.host_field].nunique() / 2), 2)
    x_ = df_indiv[it.tick_field].unique()
    i = 0
    for n, data_n in df_indiv.groupby(
            df_indiv[it.host_field]):
        agglo_containers = pd.Series(
            data=[0.0] * df_indiv[it.tick_field].nunique(), index=x_)
        ax_node_usage = fig_node_usage.add_subplot(
            gs_node_usage[int(i / 2), int(i % 2)])
        # ax_node_usage.set_title('Node %d' % dict_id_n[n)
        ax_node_usage.set(xlabel='time (s)', ylabel=metric)
        ax_node_usage.grid()

        for c, data_c in data_n.groupby(data_n[it.indiv_field]):
            c_int = [k for k, v in dict_id_c.items() if v == c][0]
            temp_df = data_c.reset_index(level=it.indiv_field, drop=True)
            agglo_containers = agglo_containers.add(temp_df[metric])
            if labels_[c_int] == to_filter:
                ax_node_usage.plot(
                    x_, agglo_containers, colors[labels_[c_int]],
                    alpha=0.0)
            else:
                ax_node_usage.plot(
                    x_, agglo_containers, colors[labels_[c_int]], label=c_int)
        # ax_node_usage.legend()
        i += 1

    plt.draw()
    return fig_node_usage


def plot_clustering_containers_by_node_spec_cont(
        df_indiv: pd.DataFrame, dict_id_c: Dict, labels_: List,
        containers_toshow: List, metric: str = 'cpu'):
    """
    Plot containers consumption grouped by node, one container added above
    another, with their cluster color.
    """
    fig_node_usage = plt.figure()
    fig_node_usage.suptitle(
        metric + ' consumption in each node, containers clustering')
    gs_node_usage = gridspec.GridSpec(math.ceil(
        df_indiv[it.host_field].nunique() / 2), 2)
    x_ = df_indiv[it.tick_field].unique()
    i = 0
    for n, data_n in df_indiv.groupby(
            df_indiv[it.host_field]):
        agglo_containers = pd.Series(
            data=[0.0] * df_indiv[it.tick_field].nunique(), index=x_)
        ax_node_usage = fig_node_usage.add_subplot(
            gs_node_usage[int(i / 2), int(i % 2)])
        # ax_node_usage.set_title('Node %d' % dict_id_n[n)
        ax_node_usage.set(xlabel='time (s)', ylabel=metric)
        ax_node_usage.grid()

        for c, data_c in data_n.groupby(data_n[it.indiv_field]):
            c_int = [k for k, v in dict_id_c.items() if v == c][0]
            temp_df = data_c.reset_index(level=it.indiv_field, drop=True)
            agglo_containers = agglo_containers.add(temp_df[metric])
            if c_int in containers_toshow:
                ax_node_usage.plot(
                    x_, agglo_containers,
                    other_colors[labels_[c_int]], label=c_int)
            else:
                ax_node_usage.plot(
                    x_, agglo_containers, colors[labels_[c_int]])
        ax_node_usage.legend()
        i += 1

    plt.draw()


def plot_containers_groupby_nodes(df_indiv: pd.DataFrame,
                                  max_cap: int, sep_time: int,
                                  title: str = None, metrics: List[str] = None) -> plt.Figure:
    """Plot containers consumption grouped by node."""
    metrics = metrics or it.metrics
    title = title or 'Node CPU consumption'
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.set_ylim([0, max_cap + (max_cap * 0.2)])

    pvt_cpu = pd.pivot_table(df_indiv, columns=it.host_field,
                             index=df_indiv[it.tick_field],
                             aggfunc='sum', values=metrics[0])
    pvt_cpu.plot(ax=ax, legend=False)
    ax.axvline(x=sep_time, color='red', linestyle='--')
    ax.axhline(y=max_cap, color='red')

    plt.draw()
    return fig


def plot_containers_groupby_nodes_px(df_indiv: pd.DataFrame,
                                     max_cap: int, sep_time: int,
                                     metrics: List[str] = None):
    """Plot containers consumption grouped by node."""
    # TODO several metrics ?
    # TODO line_shape in params
    metrics = metrics or it.metrics
    pvt = pd.pivot_table(df_indiv, columns=it.host_field,
                         index=df_indiv[it.tick_field],
                         aggfunc='sum', values=metrics[0])
    fig = px.line(pvt, line_shape='spline',
                  title='Host CPU consumption')
    fig.add_hline(max_cap, line={'color': 'red'})
    fig.add_vline(sep_time, line={'color': 'red', 'dash': 'dashdot'})
    fig.show(it.renderer)


def plot_dendrogram(z_all: np.array, k: int):
    """Plot dendrogram for the hierarchical clustering building."""
    plt.figure()
    plt.title('Hierarchical Clustering Dendrogram -- ALL')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hac.dendrogram(
        z_all,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.draw()


def plot_cluster_profiles(profiles_: List):
    """Plot mean profiles of clusters."""
    fig, ax = plt.subplots()
    fig.suptitle('Cluster profiles (mean of containers in it)')
    k = len(profiles_)

    for i in range(k):
        ax.plot(profiles_[i, :], color=colors[i], label=i)
    ax.legend()
    plt.draw()


def plot_nodes_wout_containers(instance: Instance):
    """Plot nodes consumption without containers."""
    fig, ax = plt.subplots()
    fig.suptitle('Nodes usage without containers')

    print(instance.df_host)
    np_nodes = pd.pivot_table(
        instance.df_host,
        columns=instance.df_host[it.host_field],
        index=instance.df_host[it.tick_field],
        aggfunc='sum', values='cpu')
    np_nodes_containers = pd.pivot_table(
        instance.df_indiv,
        columns=instance.df_indiv[it.host_field],
        index=instance.df_indiv[it.tick_field],
        aggfunc='sum',
        values='cpu')

    print(np_nodes)
    print(np_nodes_containers)
    new_np_nodes = np_nodes - np_nodes_containers
    print(new_np_nodes)
    new_np_nodes.plot(ax=ax)
    plt.draw()
    fig, ax = plt.subplots()
    fig.suptitle('Nodes usage with containers')
    np_nodes.plot(ax=ax)
    plt.draw()
    fig, ax = plt.subplots()
    fig.suptitle('Nodes usage from containers')
    np_nodes_containers.plot(ax=ax)
    plt.show()


def init_containers_plot(df_indiv: pd.DataFrame,
                         sep_time: int, metric: str = 'cpu'):
    """Initialize containers consumption plot."""
    fig, ax = plt.subplots()
    fig.suptitle('Containers consumption evolution')
    ax.set_ylim([0, df_indiv['cpu'].max()])
    ax.set_xlim([0, df_indiv[it.tick_field].max()])

    pvt = pd.pivot_table(
        df_indiv.loc[
            df_indiv[it.tick_field] <= sep_time],
        columns=df_indiv[it.indiv_field],
        index=df_indiv[it.tick_field], aggfunc='sum',
        values='cpu')
    ax.plot(pvt)
    ax.axvline(x=sep_time, color='red', linestyle='--')

    return (fig, ax)


def update_containers_plot(fig, ax, df: pd.DataFrame, t: int):
    """Update containers consumption plot with new data."""
    pvt_cpu = pd.pivot_table(
        df, columns=df[it.indiv_field],
        index=df[it.tick_field], aggfunc='sum', values='cpu')
    ax.plot(pvt_cpu)
    plt.pause(0.5)


def init_nodes_plot(df_indiv: pd.DataFrame, dict_id_n: Dict, sep_time: int,
                    max_cap: int, metric: str = None):
    """Initialize nodes consumption plot."""
    metric = metric or it.metrics[0]
    fig, ax = plt.subplots()
    fig.suptitle('Nodes consumption evolution')
    ax.set_xlim([0, df_indiv[it.tick_field].max()])
    ax.set_ylim([0, max_cap + (max_cap * 0.2)])
    df = df_indiv.loc[
        df_indiv[it.tick_field] <= sep_time]
    df.reset_index(drop=True, inplace=True)
    for n, data_n in df.groupby(
            df[it.host_field]):
        n_int = [k for k, v in dict_id_n.items() if v == n][0] % len(colors)
        ax.plot(data_n.groupby(data_n[it.tick_field])[metric].sum(), color=colors[n_int])

    # pvt = pd.pivot_table(
    #     df_indiv.loc[
    #         df_indiv[it.tick_field] <= sep_time],
    #     columns=it.host_field,
    #     index=df_indiv[it.tick_field],
    #     aggfunc='sum', values=metric)
    # ax.plot(pvt)
    ax.axvline(x=sep_time, color='red', linestyle='--')
    ax.axhline(y=max_cap, color='red')

    return (fig, ax)


def init_nodes_plot_px(df_indiv: pd.DataFrame, sep_time: int,
                       max_cap: int, metric: str = None):
    """Initialize nodes consumption plot."""
    metric = metric or it.metrics[0]
    pvt = pd.pivot_table(
        df_indiv.loc[
            df_indiv[it.tick_field] <= sep_time],
        columns=it.host_field,
        index=df_indiv[it.tick_field],
        aggfunc='sum', values=metric)
    fig = px.line(pvt, line_shape='spline',
                  title='Host CPU consumption')
    fig.add_hline(max_cap, line={'color': 'red'})
    fig.add_vline(sep_time, line={'color': 'red', 'dash': 'dashdot'})
    fig.show(it.renderer)
    return fig


def update_nodes_plot(fig, ax, df: pd.DataFrame,
                      dict_id_n: Dict, metric: str = None):
    """Update nodes consumption plot with new data."""
    metric = metric or it.metrics[0]
    temp_df = df.reset_index(drop=True)
    for n, data_n in temp_df.groupby(
            temp_df[it.host_field]):
        n_int = [k for k, v in dict_id_n.items() if v == n][0] % len(colors)
        ax.plot(data_n.groupby(data_n[it.tick_field])[metric].sum(), color=colors[n_int])


def update_nodes_plot_px(fig, df: pd.DataFrame, metric: str = None):
    """Update nodes consumption plot with new data."""
    # TODO not open new tab each loop...
    metric = metric or it.metrics[0]
    pvt = pd.pivot_table(
        df, columns=df[it.host_field],
        index=df[it.tick_field], aggfunc='sum', values=metric)
    for i, col in enumerate(fig.data):
        fig.data[i]['y'] = pvt[pvt.columns[i]]
        fig.data[i]['x'] = pvt.index
    fig.show(it.renderer)


def init_plot_clustering(df_clust: pd.DataFrame,
                         metric: str = None):
    """Initialize clustering plot."""
    # TODO add title, same scale, smooth curves..
    metric = metric or it.metrics[0]
    fig, ax = plt.subplots()
    fig.suptitle('Clustering evolution')
    for row in df_clust.iterrows():
        cluster = int(row[1]['cluster'])
        values = row[1].drop(labels='cluster')
        # ax.plot(values, colors[cluster], label=cluster)
        ax.plot(values, colors[cluster])
    ax.axvline(x=df_clust.columns[-2], color='red', linestyle='--')
    # ax.legend()
    return (fig, ax)


def init_plot_cluster_profiles(profiles: np.array,
                               metric: str = None):
    """Initialize clusters mean profiles plot."""
    # TODO add title, same scale, smooth curves..
    metric = metric or it.metrics[0]
    fig, ax = plt.subplots()
    fig.suptitle('Clusters mean profiles evolution')
    for i in range(len(profiles)):
        ax.plot(profiles[i], colors[i])
    return (fig, ax)


def init_plot_clustering_axes(df_clust: pd.DataFrame, dict_id_c: Dict,
                              metric: str = 'cpu'):
    """Initialize clustering plot."""
    # TODO add title, same scale, smooth curves..
    fig = plt.figure()
    fig.suptitle('Clustering evolution')
    gs = gridspec.GridSpec(df_clust.cluster.max() + 1, 1)
    ax_ = []
    for k, data in df_clust.groupby(['cluster']):
        ax_.append(fig.add_subplot(gs[k, 0]))
        ax_[k].set_title('Cluster n°%d' % k, pad=k)
        ax_[k].set(xlabel='time (s)', ylabel=metric)
        # ax_[k].grid()
        for row in data.drop(labels='cluster', axis=1).iterrows():
            c_int = [k for k, v in dict_id_c.items() if v == row[0]][0]
            # ax_[k].plot(row[1], colors[k], label=c_int)
            ax_[k].plot(row[1], label=c_int)
        ax_[k].legend()
    return (fig, ax_)


def init_plot_clustering_px(df_clust: pd.DataFrame, dict_id_c: Dict,
                            metric: str = None):
    """Initialize clustering plot."""
    metric = metric or it.metrics[0]
    fig = make_subplots(rows=int(df_clust.cluster.max() + 1),
                        shared_xaxes=True, cols=1)
    for k, data in df_clust.groupby(['cluster']):
        for row in data.drop(labels='cluster', axis=1).iterrows():
            fig.append_trace(go.Scatter(
                x=row[1].index,
                y=row[1].values,
                name=row[0]
            ), row=k + 1, col=1)
    fig.show(it.renderer)
    return fig


def update_clustering_plot(fig, ax, df_clust: pd.DataFrame,
                           dict_id_c: Dict, metric: str = None):
    """Update clustering plot with new data."""
    metric = metric or it.metrics[0]
    for row in df_clust.iterrows():
        cluster = int(row[1]['cluster'])
        values = row[1].drop(labels='cluster')
        # ax.plot(values, colors[cluster], label=row[0])
        ax.plot(values, colors[cluster])
    ax.axvline(x=df_clust.columns[-2], color='red', linestyle='--')


def update_cluster_profiles(fig, ax, profiles: np.array,
                            x: np.array, metric: str = None):
    """Update the clusters profiles."""
    metric = metric or it.metrics[0]
    for i in range(len(profiles)):
        ax.plot(x, profiles[i], colors[i])


def update_clustering_plot_axes(fig, ax_,
                                df_clust: pd.DataFrame, dict_id_c: Dict,
                                metric: str = 'cpu'):
    """Update clustering plot with new data."""
    for k, data in df_clust.groupby(['cluster']):
        for row in data.drop(labels='cluster', axis=1).iterrows():
            c_int = [k for k, v in dict_id_c.items() if v == row[0]][0]
            ax_[k].plot(row[1], colors[k], label=c_int)


def update_clustering_plot_px(
        fig, df_clust: pd.DataFrame, dict_id_c: Dict, tick: int = 1,
        metric: str = None):
    """Update clustering plot with new data."""
    pass
    # for k, data in df_clust.groupby(['cluster']):
    # for row in data.drop(labels='cluster', axis=1).iterrows():

    # for i, col in enumerate(fig.data):
    # c = fig.data[i]['name']
    # c_df = df_clust.loc[c]
    # c_clust = df_clust.loc[c, 'cluster']
    # c_df = c_df.drop(labels='cluster', axis=0)
    # print(fig.data[i], i)
    # if fig.data[i]['row'] != c_clust:
    #     fig.data[i]['row'] = c_clust
    #     print('je change')

    # fig.data[i]['y'] = np.append(fig.data[i]['y'], c_df.values[-tick:])
    # fig.data[i]['x'] = np.append(fig.data[i]['x'], c_df.index[-tick:])

    # fig.show(it.renderer)

    # def plot_nodes_adding_containers(df_indiv: pd.DataFrame,
    #                                   max_cap: int, sep_time: int):
    #     """Plot nodes consumption showing allocated containers."""
    #     fig, ax = plt.subplots()
    #     fig.suptitle('Node CPU consumption')
    #     # TODO generic
    #     ax.set_ylim([0, max_cap + (max_cap * 0.2)])


def plot_conflict_graph(graph: nx.Graph):
    """Plot the conflict graph from dual values."""
    fig, ax = plt.subplots()
    fig.suptitle('Conflict graph')
    pos = nx.spring_layout(graph, k=0.15, iterations=20)
    nx.draw(graph, pos, with_labels=True)
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'))

    fig.savefig('graph.svg')
    # plt.draw()
