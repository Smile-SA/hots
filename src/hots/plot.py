"""
Provide all plotting methods : different clustering results views,
containers data, nodes data, continous plot in evaluation step.
"""

import math

from matplotlib import gridspec as gridspec
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt

import networkx as nx

import numpy as np

import pandas as pd

import scipy.cluster.hierarchy as hac

from . import init as it

# Global variables
# TODO manage big number of colors
colors = ['blue', 'orange', 'green', 'red', 'purple',
          'brown', 'pink', 'gray', 'olive', 'cyan', 'turquoise',
          'chocolate', 'navy', 'lightcoral', 'violet']

other_colors = ['violet', 'lightcoral', 'navy', 'chocolate', 'turquoise']


# TODO better plots (especially "live" plots)
# Functions definitions #


def plot_clustering(df_clust, dict_id_c, metric=None, title=None):
    """Plot metric containers consumption, grouped by cluster.

    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    :param metric: _description_, defaults to None
    :type metric: str, optional
    :param title: _description_, defaults to None
    :type title: str, optional
    :return: _description_
    :rtype: plt.Figure
    """
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


def plot_clustering_spec_cont(
    df_clust, dict_id_c, containers_toshow, metric='cpu', title=None
):
    """Plot metric containers consumption, grouped by cluster.

    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    :param containers_toshow: _description_
    :type containers_toshow: List
    :param metric: _description_, defaults to 'cpu'
    :type metric: str, optional
    :param title: _description_, defaults to None
    :type title: str, optional
    """
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


def plot_containers_clustering_together(df_clust, metric='cpu'):
    """Plot all containers consumption with their cluster color.

    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :param metric: _description_, defaults to 'cpu'
    :type metric: str, optional
    """
    fig, ax = plt.subplots()
    fig.suptitle('Containers clustering (' + metric + ')')
    for row in df_clust.iterrows():
        ax.plot(row[1].drop(labels='cluster'),
                color=colors[int(row[1]['cluster'])])
    # ax.legend()
    plt.draw()


def plot_clustering_containers_by_node(
    df_indiv, dict_id_c, labels_, filter_big=False, metric=None
):
    """Plot containers consumption grouped by node, one container added above
    another, with their cluster color.

    :param df_indiv: _description_
    :type df_indiv: pd.DataFrame
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    :param labels_: _description_
    :type labels_: List
    :param filter_big: _description_, defaults to False
    :type filter_big: bool, optional
    :param metric: _description_, defaults to None
    :type metric: str, optional
    :return: _description_
    :rtype: plt.Figure
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
    df_indiv, dict_id_c, labels_, containers_toshow, metric='cpu'
):
    """Plot containers consumption grouped by node, one container added above
    another, with their cluster color.

    :param df_indiv: _description_
    :type df_indiv: pd.DataFrame
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    :param labels_: _description_
    :type labels_: List
    :param containers_toshow: _description_
    :type containers_toshow: List
    :param metric: _description_, defaults to 'cpu'
    :type metric: str, optional
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


def plot_containers_groupby_nodes(
    df_indiv, max_cap, sep_time, title=None, metrics=None
):
    """Plot containers consumption grouped by node.

    :param df_indiv: _description_
    :type df_indiv: pd.DataFrame
    :param max_cap: _description_
    :type max_cap: int
    :param sep_time: _description_
    :type sep_time: int
    :param title: _description_, defaults to None
    :type title: str, optional
    :param metrics: _description_, defaults to None
    :type metrics: List[str], optional
    :return: _description_
    :rtype: plt.Figure
    """
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


# The following uses plotly : comment for the moment
# def plot_containers_groupby_nodes_px(
#     df_indiv, max_cap, sep_time, metrics=None
# ):
#     """Plot containers consumption grouped by node.

#     :param df_indiv: _description_
#     :type df_indiv: pd.DataFrame
#     :param max_cap: _description_
#     :type max_cap: int
#     :param sep_time: _description_
#     :type sep_time: int
#     :param metrics: _description_, defaults to None
#     :type metrics: List[str], optional
#     """
#     # TODO several metrics ?
#     # TODO line_shape in params
#     metrics = metrics or it.metrics
#     pvt = pd.pivot_table(df_indiv, columns=it.host_field,
#                          index=df_indiv[it.tick_field],
#                          aggfunc='sum', values=metrics[0])
#     fig = px.line(pvt, line_shape='spline',
#                   title='Host CPU consumption')
#     fig.add_hline(max_cap, line={'color': 'red'})
#     fig.add_vline(sep_time, line={'color': 'red', 'dash': 'dashdot'})
#     fig.show(it.renderer)


def plot_dendrogram(z_all, k):
    """Plot dendrogram for the hierarchical clustering building.

    :param z_all: _description_
    :type z_all: np.array
    :param k: _description_
    :type k: int
    """
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


def plot_cluster_profiles(profiles_):
    """Plot mean profiles of clusters.

    :param profiles_: _description_
    :type profiles_: List
    """
    fig, ax = plt.subplots()
    fig.suptitle('Cluster profiles (mean of containers in it)')
    k = len(profiles_)

    for i in range(k):
        ax.plot(profiles_[i, :], color=colors[i], label=i)
    ax.legend()
    plt.draw()


def plot_nodes_wout_containers(instance):
    """Plot nodes consumption without containers.

    :param instance: _description_
    :type instance: Instance
    """
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


def init_containers_plot(df_indiv, sep_time, metric='cpu'):
    """Initialize containers consumption plot.

    :param df_indiv: _description_
    :type df_indiv: pd.DataFrame
    :param sep_time: _description_
    :type sep_time: int
    :param metric: _description_, defaults to 'cpu'
    :type metric: str, optional
    :return: _description_
    :rtype: Tuple
    """
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


def update_containers_plot(fig, ax, df, t):
    """Update containers consumption plot with new data.

    :param fig: _description_
    :type fig: _type_
    :param ax: _description_
    :type ax: _type_
    :param df: _description_
    :type df: pd.DataFrame
    :param t: _description_
    :type t: int
    """
    pvt_cpu = pd.pivot_table(
        df, columns=df[it.indiv_field],
        index=df[it.tick_field], aggfunc='sum', values='cpu')
    ax.plot(pvt_cpu)
    plt.pause(0.5)


def init_nodes_plot(
    df_indiv, dict_id_n, sep_time, max_cap, metric=None
):
    """Initialize nodes consumption plot.

    :param df_indiv: _description_
    :type df_indiv: pd.DataFrame
    :param dict_id_n: _description_
    :type dict_id_n: Dict
    :param sep_time: _description_
    :type sep_time: int
    :param max_cap: _description_
    :type max_cap: int
    :param metric: _description_, defaults to None
    :type metric: str, optional
    :return: _description_
    :rtype: Tuple
    """
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


# The following uses plotly : comment for the moment
# def init_nodes_plot_px(df_indiv, sep_time, max_cap, metric=None):
#     """Initialize nodes consumption plot.

#     :param df_indiv: _description_
#     :type df_indiv: pd.DataFrame
#     :param sep_time: _description_
#     :type sep_time: int
#     :param max_cap: _description_
#     :type max_cap: int
#     :param metric: _description_, defaults to None
#     :type metric: str, optional
#     :return: _description_
#     :rtype: plt.Figure
#     """
#     metric = metric or it.metrics[0]
#     pvt = pd.pivot_table(
#         df_indiv.loc[
#             df_indiv[it.tick_field] <= sep_time],
#         columns=it.host_field,
#         index=df_indiv[it.tick_field],
#         aggfunc='sum', values=metric)
#     fig = px.line(pvt, line_shape='spline',
#                   title='Host CPU consumption')
#     fig.add_hline(max_cap, line={'color': 'red'})
#     fig.add_vline(sep_time, line={'color': 'red', 'dash': 'dashdot'})
#     fig.show(it.renderer)
#     return fig


def update_nodes_plot(fig, ax, df, dict_id_n, metric=None):
    """Update nodes consumption plot with new data.

    :param fig: _description_
    :type fig: _type_
    :param ax: _description_
    :type ax: _type_
    :param df: _description_
    :type df: pd.DataFrame
    :param dict_id_n: _description_
    :type dict_id_n: Dict
    :param metric: _description_, defaults to None
    :type metric: str, optional
    """
    metric = metric or it.metrics[0]
    temp_df = df.reset_index(drop=True)
    for n, data_n in temp_df.groupby(
            temp_df[it.host_field]):
        n_int = [k for k, v in dict_id_n.items() if v == n][0] % len(colors)
        ax.plot(data_n.groupby(data_n[it.tick_field])[metric].sum(), color=colors[n_int])


# The following uses plotly : comment for the moment
# def update_nodes_plot_px(fig, df, metric=None):
#     """Update nodes consumption plot with new data.

#     :param fig: _description_
#     :type fig: _type_
#     :param df: _description_
#     :type df: pd.DataFrame
#     :param metric: _description_, defaults to None
#     :type metric: str, optional
#     """
#     # TODO not open new tab each loop...
#     metric = metric or it.metrics[0]
#     pvt = pd.pivot_table(
#         df, columns=df[it.host_field],
#         index=df[it.tick_field], aggfunc='sum', values=metric)
#     for i, col in enumerate(fig.data):
#         fig.data[i]['y'] = pvt[pvt.columns[i]]
#         fig.data[i]['x'] = pvt.index
#     fig.show(it.renderer)


def init_plot_clustering(df_clust, metric=None):
    """Initialize clustering plot.

    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :param metric: _description_, defaults to None
    :type metric: str, optional
    :return: _description_
    :rtype: Tuple
    """
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


def init_plot_cluster_profiles(profiles, metric=None):
    """Initialize clusters mean profiles plot.

    :param profiles: _description_
    :type profiles: np.array
    :param metric: _description_, defaults to None
    :type metric: str, optional
    :return: _description_
    :rtype: Tuple
    """
    # TODO add title, same scale, smooth curves..
    metric = metric or it.metrics[0]
    fig, ax = plt.subplots()
    fig.suptitle('Clusters mean profiles evolution')
    for i in range(len(profiles)):
        ax.plot(profiles[i], colors[i])
    return (fig, ax)


def init_plot_clustering_axes(df_clust, dict_id_c, metric='cpu'):
    """Initialize clustering plot.

    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    :param metric: _description_, defaults to 'cpu'
    :type metric: str, optional
    :return: _description_
    :rtype: Tuple
    """
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


# The following uses plotly : comment for the moment
# def init_plot_clustering_px(df_clust, dict_id_c, metric=None):
#     """Initialize clustering plot.

#     :param df_clust: _description_
#     :type df_clust: pd.DataFrame
#     :param dict_id_c: _description_
#     :type dict_id_c: Dict
#     :param metric: _description_, defaults to None
#     :type metric: str, optional
#     :return: _description_
#     :rtype: plt.Figure
#     """
#     metric = metric or it.metrics[0]
#     fig = make_subplots(rows=int(df_clust.cluster.max() + 1),
#                         shared_xaxes=True, cols=1)
#     for k, data in df_clust.groupby(['cluster']):
#         for row in data.drop(labels='cluster', axis=1).iterrows():
#             fig.append_trace(go.Scatter(
#                 x=row[1].index,
#                 y=row[1].values,
#                 name=row[0]
#             ), row=k + 1, col=1)
#     fig.show(it.renderer)
#     return fig


def update_clustering_plot(fig, ax, df_clust, dict_id_c, metric=None):
    """Update clustering plot with new data.

    :param fig: _description_
    :type fig: _type_
    :param ax: _description_
    :type ax: _type_
    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    :param metric: _description_, defaults to None
    :type metric: str, optional
    """
    metric = metric or it.metrics[0]
    for row in df_clust.iterrows():
        cluster = int(row[1]['cluster'])
        values = row[1].drop(labels='cluster')
        # ax.plot(values, colors[cluster], label=row[0])
        ax.plot(values, colors[cluster])
    ax.axvline(x=df_clust.columns[-2], color='red', linestyle='--')


def update_cluster_profiles(fig, ax, profiles, x, metric=None):
    """Update the clusters profiles.

    :param fig: _description_
    :type fig: _type_
    :param ax: _description_
    :type ax: _type_
    :param profiles: _description_
    :type profiles: np.array
    :param x: _description_
    :type x: np.array
    :param metric: _description_, defaults to None
    :type metric: str, optional
    """
    metric = metric or it.metrics[0]
    for i in range(len(profiles)):
        ax.plot(x, profiles[i], colors[i])


def update_clustering_plot_axes(fig, ax_, df_clust, dict_id_c, metric='cpu'):
    """Update clustering plot with new data.

    :param fig: _description_
    :type fig: _type_
    :param ax_: _description_
    :type ax_: _type_
    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    :param metric: _description_, defaults to 'cpu'
    :type metric: str, optional
    """
    for k, data in df_clust.groupby(['cluster']):
        for row in data.drop(labels='cluster', axis=1).iterrows():
            c_int = [k for k, v in dict_id_c.items() if v == row[0]][0]
            ax_[k].plot(row[1], colors[k], label=c_int)


def plot_conflict_graph(graph):
    """Plot the conflict graph from dual values.

    :param graph: _description_
    :type graph: nx.Graph
    """
    fig, ax = plt.subplots()
    fig.suptitle('Conflict graph')
    pos = nx.spring_layout(graph, k=0.15, iterations=20)
    nx.draw(graph, pos, with_labels=True)
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'))

    fig.savefig('graph.svg')
    # plt.draw()
