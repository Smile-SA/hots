"""
=========
rac plot
=========

Provide all plotting methods : different clustering results views,
containers data, nodes data, continous plot in evaluation step.
"""

import math
from typing import Dict, List

from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import scipy.cluster.hierarchy as hac

from .instance import Instance

# Global variables

colors = ['blue', 'orange', 'green', 'red', 'purple',
          'brown', 'pink', 'gray', 'olive', 'cyan', 'turquoise',
          'chocolate', 'navy', 'lightcoral', 'violet']

# TODO factorize x_ (function init in all files ?)

# Functions definitions #


def plot_clustering(df_clust: pd.DataFrame, metric: str = 'cpu',
                    title: str = None):
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
            ax_[k].plot(row[1], colors[k], label=row[0])
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
        df_containers: pd.DataFrame, dict_id_c: Dict, labels_: List,
        metric: str = 'cpu'):
    """
    Plot containers consumption grouped by node, one container added above
    another, with their cluster color.
    """
    fig_node_usage = plt.figure()
    fig_node_usage.suptitle(
        metric + ' consumption in each node, containers clustering')
    gs_node_usage = gridspec.GridSpec(math.ceil(
        df_containers['machine_id'].nunique() / 2), 2)
    x_ = df_containers['timestamp'].unique()
    i = 0

    for n, data_n in df_containers.groupby(
            df_containers['machine_id']):
        agglo_containers = pd.Series(
            data=[0.0] * df_containers['timestamp'].nunique(), index=x_)
        ax_node_usage = fig_node_usage.add_subplot(
            gs_node_usage[int(i / 2), int(i % 2)])
        # ax_node_usage.set_title('Node %d' % dict_id_n[n)
        ax_node_usage.set(xlabel='time (s)', ylabel=metric)
        ax_node_usage.grid()

        for c, data_c in data_n.groupby(data_n['container_id']):
            c_int = [k for k, v in dict_id_c.items() if v == c][0]
            temp_df = data_c.reset_index(level='container_id', drop=True)
            agglo_containers = agglo_containers.add(temp_df[metric])
            ax_node_usage.plot(
                x_, agglo_containers, colors[labels_[c_int]], label=c_int)
        ax_node_usage.legend()
        i += 1

    plt.draw()


def plot_containers_groupby_nodes(df_containers: pd.DataFrame,
                                  max_cap: int, sep_time: int):
    # TODO make metrics generic
    """Plot containers consumption grouped by node."""
    # print("Preparing plot containers grouped by nodes ...")
    # fig = plt.figure()
    # fig.suptitle("Containers CPU & mem consumption grouped by node")
    # gs = gridspec.GridSpec(2, 1)

    # ax_cpu = fig.add_subplot(gs[0, 0])
    # ax_mem = fig.add_subplot(gs[1, 0])
    # pvt_cpu = pd.pivot_table(df_container, columns="machine_id",
    #                          index=df_container["timestamp"],
    # aggfunc="sum", values="cpu")
    # pvt_mem = pd.pivot_table(df_container, columns="machine_id",
    #                          index=df_container["timestamp"],
    # aggfunc="sum", values="mem")

    # pvt_cpu.plot(ax=ax_cpu)
    # pvt_mem.plot(ax=ax_mem)

    fig, ax = plt.subplots()
    fig.suptitle('Node CPU consumption')
    # TODO generic
    ax.set_ylim([0, max_cap + (max_cap * 0.2)])

    pvt_cpu = pd.pivot_table(df_containers, columns='machine_id',
                             index=df_containers['timestamp'],
                             aggfunc='sum', values='cpu')
    pvt_cpu.plot(ax=ax, legend=False)
    ax.axvline(x=sep_time, color='red', linestyle='--')
    ax.axhline(y=max_cap, color='red')

    plt.draw()


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

    print(instance.df_nodes)
    np_nodes = pd.pivot_table(
        instance.df_nodes,
        columns=instance.df_nodes['machine_id'],
        index=instance.df_nodes['timestamp'],
        aggfunc='sum', values='cpu')
    np_nodes_containers = pd.pivot_table(
        instance.df_containers,
        columns=instance.df_containers['machine_id'],
        index=instance.df_containers['timestamp'],
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


def init_containers_plot(df_containers: pd.DataFrame,
                         sep_time: int, metric: str = 'cpu'):
    """Initialize containers consumption plot."""
    fig, ax = plt.subplots()
    fig.suptitle('Containers consumption evolution')
    ax.set_ylim([0, df_containers['cpu'].max()])
    ax.set_xlim([0, df_containers['timestamp'].max()])

    pvt = pd.pivot_table(
        df_containers.loc[
            df_containers['timestamp'] <= sep_time],
        columns=df_containers['container_id'],
        index=df_containers['timestamp'], aggfunc='sum',
        values='cpu')
    ax.plot(pvt)
    ax.axvline(x=sep_time, color='red', linestyle='--')

    return (fig, ax)


def update_containers_plot(fig, ax, df: pd.DataFrame, t: int):
    """Update containers consumption plot with new data."""
    pvt_cpu = pd.pivot_table(
        df, columns=df['container_id'],
        index=df['timestamp'], aggfunc='sum', values='cpu')
    ax.plot(pvt_cpu)
    plt.pause(0.5)


# TODO set generic lim + hline
def init_nodes_plot(df_containers: pd.DataFrame,
                    sep_time: int, metric: str = 'cpu'):
    """Initialize nodes consumption plot."""
    fig, ax = plt.subplots()
    fig.suptitle('Nodes consumption evolution')
    ax.set_xlim([0, df_containers['timestamp'].max()])
    ax.set_ylim([0, 20])

    pvt = pd.pivot_table(
        df_containers.loc[
            df_containers['timestamp'] <= sep_time],
        columns='machine_id',
        index=df_containers['timestamp'],
        aggfunc='sum', values='cpu')
    ax.plot(pvt)
    ax.axvline(x=sep_time, color='red', linestyle='--')
    ax.axhline(y=20, color='red')

    return (fig, ax)


def update_nodes_plot(fig, ax, df: pd.DataFrame, t):
    """Update nodes consumption plot with new data."""
    pvt_cpu = pd.pivot_table(
        df, columns=df['machine_id'],
        index=df['timestamp'], aggfunc='sum', values='cpu')
    ax.plot(pvt_cpu)
    plt.pause(0.5)


def init_plot_clustering(df_clust: pd.DataFrame, metric: str = 'cpu'):
    """Initialize clustering plot."""
    fig = plt.figure()
    fig.suptitle('Clustering evolution')
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
            ax_[k].plot(row[1], colors[k], label=row[0])
        ax_[k].legend()
    return (fig, ax_)


def update_clustering_plot(fig, ax_,
                           df_clust: pd.DataFrame, metric: str = 'cpu'):
    """Update clustering plot with new data."""
    for k, data in df_clust.groupby(['cluster']):
        for row in data.drop(labels='cluster', axis=1).iterrows():
            ax_[k].plot(row[1], colors[k], label=row[0])


# def plot_nodes_adding_containers(df_containers: pd.DataFrame,
#                                   max_cap: int, sep_time: int):
#     """Plot nodes consumption showing allocated containers."""
#     fig, ax = plt.subplots()
#     fig.suptitle('Node CPU consumption')
#     # TODO generic
#     ax.set_ylim([0, max_cap + (max_cap * 0.2)])
