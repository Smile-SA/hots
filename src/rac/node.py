# coding=utf-8

# print(__doc__)

import math
from typing import Dict

from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from tqdm import tqdm

from .init import metrics


# Definition of Node-related functions #


def plot_specificData_allNodes(df_nodes: pd.DataFrame, metric: str):
    # TODO create temp df is bad...
    fig, ax = plt.subplots()
    temp_df = df_nodes.reset_index(drop=True)
    pvt = pd.pivot_table(temp_df, columns='machine_id',
                         index='timestamp', aggfunc='sum', values=metric)
    pvt.plot(ax=ax)
    fig.suptitle('%s use on all nodes' % metric)
    plt.draw()


def plot_allData_allNodes_end(df_nodes: pd.DataFrame, total_time: int):
    fig = plt.figure()
    gs = gridspec.GridSpec(len(metrics), 1)
    filtered_df = df_nodes.loc[df_nodes['timestamp']
                               > total_time / 2]
    x_ = filtered_df['timestamp'].unique()
    ax_ = []
    for ai in range(len(metrics)):
        ax_.append(fig.add_subplot(gs[ai, 0]))
        ax_[ai].set_title(metrics[ai])
        ax_[ai].set(xlabel='time (s)', ylabel=metrics[ai])
        ax_[ai].grid()
    for key, data in filtered_df.groupby(filtered_df['machine_id']):
        ai = 0
        for metric in metrics:
            ax_[ai].plot(x_, data[metric], label=key)
            ai += 1
    fig.suptitle('Resource usage on all nodes in 2nd period')
    fig.align_labels()

    plt.draw()


def plot_total_usage(df_nodes: pd.DataFrame,
                     title: str = 'Total conso on all nodes'):
    """
    Plot the global resources consumption and return the global
    maximum usage for each metric.
    """
    temp_df = df_nodes.reset_index(level='machine_id', drop=True)
    usage_all_nodes = pd.DataFrame(
        data=0.0, columns=df_nodes.timestamp.unique(), index=metrics)
    separation_time = math.ceil(df_nodes.timestamp.nunique() / 2)
    pbar = tqdm(temp_df.groupby(temp_df['machine_id']))
    pbar.set_description('Compute total usage')
    for key, data in pbar:
        usage_all_nodes = usage_all_nodes.add(data[metrics].T)

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


def get_mean_consumption(df_nodes: pd.DataFrame):
    """
    Compute the mean consumption for each metric in each node and globally.
    """
    for metric in metrics:
        global_mean = 0.0
        print('Mean ', metric)
        for key, data in df_nodes.groupby(df_nodes['machine_id']):
            global_mean += float(data[metric].mean())
        print('Global mean : ', float(
            global_mean / df_nodes.machine_id.nunique()))


def get_list_mean(df_nodes: pd.DataFrame, total_time: int) -> (Dict, Dict):
    """Return list of mean for each metric in each node."""
    dict_mean_cpu = dict()
    dict_mean_mem = dict()

    for key, data in df_nodes.groupby(df_nodes['machine_id']):
        dict_mean_cpu[key] = float(data['cpu'].mean())
        dict_mean_mem[key] = float(data['mem'].mean())

    return (dict_mean_cpu, dict_mean_mem)


def get_list_var(df_nodes: pd.DataFrame, total_time: int) -> (Dict, Dict):
    """Return list of variance for each metric in each node."""
    dict_var_cpu = dict()
    dict_var_mem = dict()

    for key, data in df_nodes.groupby(df_nodes['machine_id']):
        dict_var_cpu[key] = float(data['cpu'].var())
        dict_var_mem[key] = float(data['mem'].var())

    return (dict_var_cpu, dict_var_mem)


def get_variance_consumption(df_nodes: pd.DataFrame):
    """
    Compute the variance and standard deviation consumption for each metric
    in each node and globally.
    """
    for metric in metrics:
        global_variance = 0.0
        global_stand_deviation = 0.0
        print('Variance & Standard deviation', metric)
        for key, data in df_nodes.groupby(df_nodes['machine_id']):
            global_variance += float(data[metric].var())
            global_stand_deviation += math.sqrt(
                float(data[metric].var()))
        print('Global Variance : ', float(
            global_variance / df_nodes.machine_id.nunique()))
        print('Global Standard deviation : ',
              float(global_stand_deviation / df_nodes.machine_id.nunique()))


def print_vmr(df_nodes: pd.DataFrame, total_time: int, part: int):
    """
    Compute the VMR (Variance-to-mean ratio) for each metric in each node.
    """
    for metric in metrics:
        global_mean = 0.0
        global_var = 0.0
        global_vmr = 0.0
        print('Mean, variance and VMR for ', metric)
        for key, data in df_nodes.groupby(df_nodes['machine_id']):
            if part == 1:
                mean = float(data.loc[data.timestamp <=
                                      total_time / 2, [metric]].mean())
                var = float(data.loc[data.timestamp <=
                                     total_time / 2, [metric]].var())
            else:
                mean = float(data.loc[data.timestamp >
                                      total_time / 2, [metric]].mean())
                var = float(data.loc[data.timestamp >
                                     total_time / 2, [metric]].var())
            print(mean, var, (var / mean))
            global_mean += mean
            global_var += var
            global_vmr += (var / mean)

        print('Global mean : ', float(
            (global_mean / df_nodes.machine_id.nunique())))
        print('Global var : ', float(
            global_var / df_nodes.machine_id.nunique()))
        print('Global vmr : ', float(
            global_vmr / df_nodes.machine_id.nunique()))


def get_list_vmr(df_nodes: pd.DataFrame, total_time: int) -> (Dict, Dict):
    """
    Return list of VMR (Variance-to-Mean Ratio) for each metric in each node.
    """
    dict_vmr_cpu = dict()
    dict_vmr_mem = dict()

    for key, data in df_nodes.groupby(df_nodes['machine_id']):
        mean = float(data['cpu'].mean())
        var = float(data['cpu'].var())
        dict_vmr_cpu[key] = (var / mean)
        mean = float(data['mem'].mean())
        var = float(data['mem'].var())
        dict_vmr_mem[key] = (var / mean)

    return (dict_vmr_cpu, dict_vmr_mem)


def get_nodes_variance(
        df_nodes: pd.DataFrame, total_time: int, part: int):
    """
    Compute the Variance for each metric in each node and return the results
    in two numpy arrays.
    """
    var = np.zeros(
        (len(metrics), df_nodes['machine_id'].nunique()), dtype=float)
    global_var = np.zeros(len(metrics), dtype=float)
    m = 0
    for metric in metrics:
        n = 0
        for key, data in df_nodes.groupby(df_nodes['machine_id']):
            if part == 1:
                var[m, n] = float(data.loc[data.timestamp <=
                                           total_time / 2, [metric]].var())
            else:
                var[m, n] = float(data.loc[data.timestamp >
                                           total_time / 2, [metric]].var())
            global_var[m] += var[m, n]
            n += 1
        m += 1

    return (var, global_var)


def build_dict_id_nodes(df_nodes: pd.DataFrame) -> Dict:
    dict_id_n = dict()
    i = 0
    for key in df_nodes.machine_id.unique():
        dict_id_n[i] = key
        i += 1

    return dict_id_n


def plot_allData_allNodes(df_nodes: pd.DataFrame):
    """Plot all metrics node consumption."""
    print('Build nodes usage plot ...')
    fig = plt.figure()
    fig.suptitle('Resource usage on all nodes')
    gs = gridspec.GridSpec(len(metrics), 1)
    ax_ = []
    ai = 0
    for metric in metrics:
        ax_.append(fig.add_subplot(gs[ai, 0]))
        ax_[ai].set_title(metric)
        ax_[ai].set(xlabel='time (s)', ylabel=metric)
        ax_[ai].grid()
        temp_df = df_nodes.reset_index(drop=True)
        pvt = pd.pivot_table(temp_df, columns='machine_id',
                             index='timestamp', aggfunc='sum', values=metric)
        pvt.plot(ax=ax_[ai], legend=False)
        ai += 1

    fig.align_labels()
    plt.draw()
