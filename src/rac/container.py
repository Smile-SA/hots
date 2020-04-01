from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt

import pandas as pd


# Definition of Container-related functions #

def plot_specificData_allContainers(df_containers, dataName):
    fig, ax = plt.subplots()
    temp_df = df_containers.reset_index(drop=True)
    pvt = pd.pivot_table(temp_df, columns="container_id",
                         index="timestamp", aggfunc="sum", values=dataName)
    pvt.plot(ax=ax)
    fig.suptitle('%s use on all containers' % dataName)
    plt.draw()


def plot_allData_allContainers(df_containers, metrics=['cpu'], sep_time=72):
    """Plot all metrics container consumption."""
    print("Build containers usage plot ...")
    fig = plt.figure()
    fig.suptitle("Resource usage on all containers")
    gs = gridspec.GridSpec(len(metrics), 1)
    ax_ = []
    ai = 0
    for metric in metrics:
        ax_.append(fig.add_subplot(gs[ai, 0]))
        ax_[ai].set_title(metric)
        ax_[ai].set(xlabel='time (s)', ylabel=metric)
        ax_[ai].grid()
        temp_df = df_containers.reset_index(drop=True)
        pvt = pd.pivot_table(temp_df, columns="container_id",
                             index="timestamp", aggfunc="sum", values=metric)
        pvt.plot(ax=ax_[ai], legend=False)
        ax_[ai].axvline(x=sep_time, color='red', linestyle='--')
        ai += 1

    fig.align_labels()
    plt.draw()


def build_dict_id_containers(df_containers):
    dict_id_c = {}
    i = 0
    for key in df_containers.container_id.unique():
        dict_id_c[i] = key
        i += 1

    return dict_id_c
