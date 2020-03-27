# coding=utf-8

# print(__doc__)

import scipy.cluster.hierarchy as hac
import math
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec

# Global variables

colors = ['blue', 'orange', 'green', 'red', 'purple',
          'brown', 'pink', 'gray', 'olive', 'cyan', 'turquoise',
          'chocolate', 'navy', 'lightcoral', 'violet']

# TODO factorize x_ (function init in all files ?)

# Functions definitions #


def plot_clustering(df_clust, metric='cpu'):
    """
    Plot metric containers consumption, grouped by cluster
    """
    fig = plt.figure()
    fig.suptitle(metric + ' consumption of containers grouped by cluster')
    gs = gridspec.GridSpec(df_clust.cluster.max()+1, 1)
    ax_ = []
    max_cons = df_clust.drop(labels='cluster', axis=1).values.max()
    for k, data in df_clust.groupby(['cluster']):
        ax_.append(fig.add_subplot(gs[k, 0]))
        ax_[k].set_title('Cluster n°%d' % k, pad=k)
        ax_[k].set(xlabel='time (s)', ylabel=metric)
        ax_[k].grid()
        ax_[k].set_ylim([0, math.ceil(max_cons)])
        for row in data.drop(labels='cluster', axis=1).iterrows():
            ax_[k].plot(row[1], colors[k])
        ax_[k].legend()
    plt.draw()


def plot_containers_clustering_together(df_clust, metric='cpu'):
    """
    Plot all containers consumption with their cluster color
    """
    print("Preparing plot containers clustering together ...")
    fig, ax = plt.subplots()
    fig.suptitle('Containers clustering (' + metric + ')')
    for row in df_clust.iterrows():
        ax.plot(row[1].drop(labels='cluster'),
                color=colors[int(row[1]['cluster'])])
    # ax.legend()
    plt.draw()


def plot_clustering_containers_byNode(instance, labels_, metric='cpu'):
    """
    Plot containers consumption grouped by node, one container added above
    another, with their cluster color
    """
    print("Preparing plot containers clustering by Node ...")
    fig_node_usage = plt.figure()
    fig_node_usage.suptitle(
        metric + " consumption in each node, containers clustering")
    gs_node_usage = gridspec.GridSpec(math.ceil(instance.nb_nodes/2), 2)
    x_ = instance.df_nodes['timestamp'].unique()
    i = 0

    for n, data_n in instance.df_containers.groupby(
            instance.df_containers['machine_id']):
        agglo_containers = pd.Series(
            data=[0.0]*instance.time, index=x_)
        ax_node_usage = fig_node_usage.add_subplot(
            gs_node_usage[int(i/2), int(i % 2)])

        for c, data_c in data_n.groupby(data_n['container_id']):
            temp_df = data_c.reset_index(level='container_id', drop=True)
            agglo_containers = agglo_containers.add(temp_df[metric])
            ax_node_usage.plot(
                x_, agglo_containers, colors[labels_[c]])
        i += 1

    plt.draw()


def plot_containers_groupby_nodes(df_container):
    # TODO make metrics generic
    """
    Plot containers consumption grouped by node
    """
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
    fig.suptitle("Node CPU consumption")
    ax.set_ylim([0, 25])

    pvt_cpu = pd.pivot_table(df_container, columns="machine_id",
                             index=df_container["timestamp"],
                             aggfunc="sum", values="cpu")
    pvt_cpu.plot(ax=ax, legend=False)
    ax.axvline(x=72, color='red', linestyle='--')
    ax.axhline(y=20, color='red')

    plt.draw()


def plot_dendrogram(Z_all, k):
    plt.figure()
    plt.title('Hierarchical Clustering Dendrogram -- ALL')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hac.dendrogram(
        Z_all,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.draw()


def plot_cluster_profiles(profiles_):
    fig, ax = plt.subplots()
    fig.suptitle("Cluster profiles (mean of containers in it)")
    k = len(profiles_)

    for i in range(k):
        ax.plot(profiles_[i, :], color=colors[i], label=i)
    ax.legend()
    plt.draw()


def plot_nodes_wout_containers(instance):
    fig, ax = plt.subplots()
    fig.suptitle("Nodes usage without containers")

    print(instance.df_nodes)
    np_nodes = pd.pivot_table(
        instance.df_nodes,
        columns=instance.df_nodes["machine_id"],
        index=instance.df_nodes["timestamp"],
        aggfunc="sum", values="cpu")
    np_nodes_containers = pd.pivot_table(
        instance.df_containers,
        columns=instance.df_containers["machine_id"],
        index=instance.df_containers["timestamp"],
        aggfunc="sum",
        values="cpu")

    print(np_nodes)
    print(np_nodes_containers)
    new_np_nodes = np_nodes - np_nodes_containers
    print(new_np_nodes)
    new_np_nodes.plot(ax=ax)
    plt.draw()
    fig, ax = plt.subplots()
    fig.suptitle("Nodes usage with containers")
    np_nodes.plot(ax=ax)
    plt.draw()
    fig, ax = plt.subplots()
    fig.suptitle("Nodes usage from containers")
    np_nodes_containers.plot(ax=ax)
    plt.show()


def update_evaluation_plot(fig, ax, df, t):
    pvt_cpu = pd.pivot_table(
        df, columns=df["container_id"],
        index=df["timestamp"], aggfunc="sum", values="cpu")
    ax.plot(pvt_cpu)
    plt.pause(0.5)


def init_plot_clustering(df_clust, metric='cpu'):
    fig = plt.figure()
    fig.suptitle('Clustering evolution')
    gs = gridspec.GridSpec(df_clust.cluster.max()+1, 1)
    ax_ = []
    max_cons = df_clust.drop(labels='cluster', axis=1).values.max()
    for k, data in df_clust.groupby(['cluster']):
        ax_.append(fig.add_subplot(gs[k, 0]))
        ax_[k].set_title('Cluster n°%d' % k, pad=k)
        ax_[k].set(xlabel='time (s)', ylabel=metric)
        ax_[k].grid()
        ax_[k].set_ylim([0, math.ceil(max_cons)])
        for row in data.drop(labels='cluster', axis=1).iterrows():
            ax_[k].plot(row[1], colors[k])
        ax_[k].legend()
    return (fig, ax_)


def update_clustering_plot(fig, ax_, df_clust, metric='cpu'):
    for k, data in df_clust.groupby(['cluster']):
        for row in data.drop(labels='cluster', axis=1).iterrows():
            ax_[k].plot(row[1], colors[k])
