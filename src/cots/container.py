"""
=========
cots container
=========

Provide actions specific to containers (plot containers data,
build dictionnary for container IDs ...)
"""

from typing import List

from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt

import pandas as pd

# Definition of Container-related functions #


def plot_data_all_containers(df_containers: pd.DataFrame, metric: str):
    """Plot a specific metric containers consumption."""
    fig, ax = plt.subplots()
    temp_df = df_containers.reset_index(drop=True)
    pvt = pd.pivot_table(temp_df, columns='container_id',
                         index='timestamp', aggfunc='sum', values=metric)
    pvt.plot(ax=ax)
    fig.suptitle('%s use on all containers' % metric)
    plt.draw()


def plot_all_data_all_containers(
        df_containers: pd.DataFrame,
        metrics: List[str] = None, sep_time: int = 72):
    """Plot all metrics containers consumption."""
    print('Build containers usage plot ...')
    metrics = metrics or ['cpu']
    fig = plt.figure()
    fig.suptitle('Resource usage on all containers')
    gs = gridspec.GridSpec(len(metrics), 1)
    ax_ = []
    ai = 0
    for metric in metrics:
        ax_.append(fig.add_subplot(gs[ai, 0]))
        ax_[ai].set_title(metric)
        ax_[ai].set(xlabel='time (s)', ylabel=metric)
        ax_[ai].grid()
        temp_df = df_containers.reset_index(drop=True)
        pvt = pd.pivot_table(temp_df, columns='container_id',
                             index='timestamp', aggfunc='sum', values=metric)
        # pvt.plot(ax=ax_[ai], legend=False)
        pvt.plot(ax=ax_[ai])
        ax_[ai].axvline(x=sep_time, color='red', linestyle='--')
        ai += 1

    fig.align_labels()
    plt.draw()


def build_dict_id_containers(df_containers: pd.DataFrame):
    """Build dictionnary for corresponding IDs and indexes."""
    dict_id_c = {}
    i = 0
    for key in df_containers.container_id.unique():
        dict_id_c[i] = key
        i += 1

    return dict_id_c
