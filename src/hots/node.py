"""
Provide actions specific to nodes (plot nodes data, build dictionnary
for node IDs, compute different statistic measures ...)
"""

import math

from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

from tqdm import tqdm

from . import init as it

# Definition of Node-related functions #


def plot_data_all_nodes(df_host, metric, max_cap, sep_time):
    """Plot specific metric consumption for all nodes.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    :param metric: _description_
    :type metric: str
    :param max_cap: _description_
    :type max_cap: float
    :param sep_time: _description_
    :type sep_time: int
    :return: _description_
    :rtype: plt.Figure
    """
    # TODO create temp df is bad...
    fig, ax = plt.subplots()
    # temp_df = df_host.reset_index(drop=True)
    ax.set_ylim([0, max_cap + (max_cap * 0.2)])
    pvt = pd.pivot_table(df_host.reset_index(drop=True), columns=it.host_field,
                         index=it.tick_field, aggfunc='sum', values=metric)
    pvt.plot(ax=ax, legend=False)
    # fig.suptitle('%s use on all nodes' % metric)
    ax.axvline(x=sep_time, color='red', linestyle='--')
    ax.axhline(y=max_cap, color='red')
    plt.draw()
    return fig


def plot_all_data_all_nodes_end(df_host, total_time):
    """Plot all metrics consumption for all nodes.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    :param total_time: _description_
    :type total_time: int
    """
    fig = plt.figure()
    gs = gridspec.GridSpec(len(it.metrics), 1)
    filtered_df = df_host.loc[
        df_host[it.tick_field] > total_time / 2]
    x_ = filtered_df[it.tick_field].unique()
    ax_ = []
    for ai in range(len(it.metrics)):
        ax_.append(fig.add_subplot(gs[ai, 0]))
        ax_[ai].set_title(it.metrics[ai])
        ax_[ai].set(xlabel='time (s)', ylabel=it.metrics[ai])
        ax_[ai].grid()
    for key, data in filtered_df.groupby(filtered_df[it.host_field]):
        ai = 0
        for metric in it.metrics:
            ax_[ai].plot(x_, data[metric], label=key)
            ai += 1
    fig.suptitle('Resource usage on all nodes in 2nd period')
    fig.align_labels()

    plt.draw()


def plot_total_usage(df_host, title='Total conso on all nodes'):
    """Plot the global resources consumption and return the global
    maximum usage for each metric.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    :param title: _description_, defaults to 'Total conso on all nodes'
    :type title: str, optional
    :return: _description_
    :rtype: Tuple[float, float]
    """
    temp_df = df_host.reset_index(level=it.host_field, drop=True)
    usage_all_nodes = pd.DataFrame(
        data=0.0, columns=df_host.timestamp.unique(), index=it.metrics)
    separation_time = math.ceil(df_host.timestamp.nunique() / 2)
    pbar = tqdm(temp_df.groupby(temp_df[it.host_field]))
    pbar.set_description('Compute total usage')
    for key, data in pbar:
        usage_all_nodes = usage_all_nodes.add(data[it.metrics].T)

    fig = plt.figure()
    fig.suptitle(title)
    gs = gridspec.GridSpec(2, 1)
    for metric in ['cpu', 'mem']:
        if metric == 'cpu':
            ax = fig.add_subplot(gs[0, 0])
        else:
            ax = fig.add_subplot(gs[1, 0])
        usage_all_nodes.loc[metric, :].plot(ax=ax)
        plt.axvline(
            x=usage_all_nodes.columns[separation_time],
            color='red', linestyle='--')
    plt.draw()

    return (usage_all_nodes.loc['cpu', :].max(),
            usage_all_nodes.loc['mem', :].max())


def get_mean_consumption(df_host):
    """Compute mean consumption for each metric in each node and globally.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    """
    for metric in it.metrics:
        global_mean = 0.0
        print('Mean ', metric)
        for key, data in df_host.groupby(df_host[it.host_field]):
            global_mean += float(data[metric].mean())
        print('Global mean : ', float(
            global_mean / df_host[it.tick_field].nunique()))


def get_list_mean(df_host, total_time):
    """Return list of mean for each metric in each node.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    :param total_time: _description_
    :type total_time: int
    :return: _description_
    :rtype: Tuple[Dict, Dict]
    """
    dict_mean_cpu = {}
    dict_mean_mem = {}

    for key, data in df_host.groupby(df_host[it.host_field]):
        dict_mean_cpu[key] = float(data['cpu'].mean())
        dict_mean_mem[key] = float(data['mem'].mean())

    return (dict_mean_cpu, dict_mean_mem)


def get_list_var(df_host, total_time):
    """Return list of variance for each metric in each node.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    :param total_time: _description_
    :type total_time: int
    :return: _description_
    :rtype: Tuple[Dict, Dict]
    """
    dict_var_cpu = {}
    dict_var_mem = {}

    for key, data in df_host.groupby(df_host[it.host_field]):
        dict_var_cpu[key] = float(data['cpu'].var())
        dict_var_mem[key] = float(data['mem'].var())

    return (dict_var_cpu, dict_var_mem)


def get_variance_consumption(df_host):
    """Compute the variance and standard deviation consumption for each metric
    in each node and globally.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    """
    for metric in it.metrics:
        global_variance = 0.0
        global_stand_deviation = 0.0
        print('Variance & Standard deviation', metric)
        for key, data in df_host.groupby(df_host[it.host_field]):
            global_variance += float(data[metric].var())
            global_stand_deviation += math.sqrt(
                float(data[metric].var()))
        print('Global Variance : ', float(
            global_variance / df_host.machine_id.nunique()))
        print('Global Standard deviation : ',
              float(global_stand_deviation / df_host.machine_id.nunique()))


def print_vmr(df_host, total_time, part):
    """Compute VMR (Variance-to-mean ratio) for each metric in each node.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    :param total_time: _description_
    :type total_time: int
    :param part: _description_
    :type part: int
    """
    for metric in it.metrics:
        global_mean = 0.0
        global_var = 0.0
        global_vmr = 0.0
        print('Mean, variance and VMR for ', metric)
        for key, data in df_host.groupby(df_host[it.host_field]):
            if part == 1:
                mean = float(data.loc[
                    data.timestamp <= total_time / 2, [metric]].mean())
                var = float(data.loc[
                    data.timestamp <= total_time / 2, [metric]].var())
            else:
                mean = float(data.loc[
                    data.timestamp > total_time / 2, [metric]].mean())
                var = float(data.loc[
                    data.timestamp > total_time / 2, [metric]].var())
            print(mean, var, (var / mean))
            global_mean += mean
            global_var += var
            global_vmr += (var / mean)

        print('Global mean : ', float(
            (global_mean / df_host.machine_id.nunique())))
        print('Global var : ', float(
            global_var / df_host.machine_id.nunique()))
        print('Global vmr : ', float(
            global_vmr / df_host.machine_id.nunique()))


def get_list_vmr(df_host, total_time):
    """Compute VMR (Variance-to-mean ratio) for each metric in each node.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    :param total_time: _description_
    :type total_time: int
    :return: _description_
    :rtype: Tuple[Dict, Dict]
    """
    dict_vmr_cpu = {}
    dict_vmr_mem = {}

    for key, data in df_host.groupby(df_host[it.host_field]):
        mean = float(data['cpu'].mean())
        var = float(data['cpu'].var())
        dict_vmr_cpu[key] = (var / mean)
        mean = float(data['mem'].mean())
        var = float(data['mem'].var())
        dict_vmr_mem[key] = (var / mean)

    return (dict_vmr_cpu, dict_vmr_mem)


def get_nodes_variance(df_host, total_time, part):
    """Compute the Variance for each metric in each node and return the results
    in two numpy arrays.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    :param total_time: _description_
    :type total_time: int
    :param part: _description_
    :type part: int
    :return: _description_
    :rtype: Tuple[np.array, np.array]
    """
    var = np.zeros(
        (len(it.metrics), df_host[it.host_field].nunique()), dtype=float)
    global_var = np.zeros(len(it.metrics), dtype=float)
    m = 0
    for metric in it.metrics:
        n = 0
        for key, data in df_host.groupby(df_host[it.host_field]):
            if part == 1:
                var[m, n] = float(data.loc[
                    data.timestamp <= total_time / 2, [metric]].var())
            else:
                var[m, n] = float(data.loc[
                    data.timestamp > total_time / 2, [metric]].var())
            global_var[m] += var[m, n]
            n += 1
        m += 1

    return (var, global_var)


def build_dict_id_nodes(df_host):
    """Build dictionnary for corresponding IDs and indexes.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    :return: _description_
    :rtype: Dict
    """
    dict_id_n = {}
    i = 0
    for key in df_host[it.host_field].unique():
        dict_id_n[i] = key
        i += 1

    return dict_id_n


def plot_all_data_all_nodes(df_host):
    """Plot all metrics node consumption.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    """
    print('Build nodes usage plot ...')
    fig = plt.figure()
    fig.suptitle('Resource usage on all nodes')
    gs = gridspec.GridSpec(len(it.metrics), 1)
    ax_ = []
    ai = 0
    for metric in it.metrics:
        ax_.append(fig.add_subplot(gs[ai, 0]))
        ax_[ai].set_title(metric)
        ax_[ai].set(xlabel='time (s)', ylabel=metric)
        ax_[ai].grid()
        temp_df = df_host.reset_index(drop=True)
        pvt = pd.pivot_table(temp_df, columns=it.host_field,
                             index=it.tick_field, aggfunc='sum', values=metric)
        pvt.plot(ax=ax_[ai], legend=False)
        ai += 1

    fig.align_labels()
    plt.draw()


def get_mean_consumption_node(df_host, node_id):
    """Get mean consumption of node_id.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    :param node_id: _description_
    :type node_id: str
    :return: _description_
    :rtype: float
    """
    return np.mean(
        df_host.loc[
            df_host[it.host_field] == node_id, ['cpu']].to_numpy()
    )


def get_nodes_load_info(df_host, df_host_meta):
    """Get all wanted node information in a dataframe.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    :param df_host_meta: _description_
    :type df_host_meta: pd.DataFrame
    :return: _description_
    :rtype: pd.DataFrame
    """
    results_df = pd.DataFrame(
        columns=[it.host_field, 'load_var', 'avg_load', 'min_load', 'max_load'])
    metric = it.metrics[0]

    for node, data_n in df_host.groupby(df_host[it.host_field]):
        node_cap = df_host_meta.loc[
            df_host_meta[it.host_field] == node
        ][metric].to_numpy()[0]
        results_df = pd.concat([
            results_df,
            pd.DataFrame.from_records([{
                it.host_field: node,
                'load_var': data_n[metric].var(),
                'avg_load': data_n[metric].mean() / node_cap * 100,
                'min_load': data_n[metric].min() / node_cap * 100,
                'max_load': data_n[metric].max() / node_cap * 100,
                'avg_cons': data_n[metric].mean(),
                'min_cons': data_n[metric].min(),
                'max_cons': data_n[metric].max()
            }])]
        )
    results_df.set_index(it.host_field, inplace=True)

    return results_df


def check_capacities(df_host, df_host_meta):
    """Check if node capacities are satisfied at a given time.

    :param df_host: _description_
    :type df_host: pd.DataFrame
    :param df_host_meta: _description_
    :type df_host_meta: pd.DataFrame
    :return: _description_
    :rtype: List
    """
    host_overload = []

    for host, host_data in df_host.groupby(it.host_field):
        if host_data[it.metrics[0]].to_numpy()[0] > df_host_meta.loc[
            df_host_meta[it.host_field] == host
        ][it.metrics[0]].to_numpy()[0]:
            host_overload.append(host)

    return host_overload
