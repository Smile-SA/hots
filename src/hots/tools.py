"""Provide external tools usefull for the code."""

import os
import sys
from pathlib import Path

import pandas as pd

import psutil

from . import init as it


def print_size_vars(list_vars):
    """Display size (memory) of variables.

    :param list_vars: _description_
    :type list_vars: List
    """
    print('List of variables with sizes:')
    for name, size in sorted(
        ((name, sys.getsizeof(value)) for name, value in list_vars),
        key=lambda x: -x[1]
    )[:10]:
        print('{:>30}: {:>8}'.format(name, sizeof_fmt(size)))


def sizeof_fmt(num, suffix='B'):
    """Ease display of memory sizes.

    :param num: _description_
    :type num: float
    :param suffix: _description_, defaults to 'B'
    :type suffix: str, optional
    :return: _description_
    :rtype: str
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return '%3.1f %s%s' % (num, unit, suffix)
        num /= 1024.0
    return '%.1f %s%s' % (num, 'Yi', suffix)


def change_max_dataset(
    df_indiv, container, col_indiv, new_max, col_val, col_time, min_time
):
    """Change dataset after max bound change.

    :param df_indiv: _description_
    :type df_indiv: pd.DataFrame
    :param container: _description_
    :type container: str
    :param col_indiv: _description_
    :type col_indiv: str
    :param new_max: _description_
    :type new_max: float
    :param col_val: _description_
    :type col_val: str
    :param col_time: _description_
    :type col_time: str
    :param min_time: _description_
    :type min_time: int
    :return: _description_
    :rtype: pd.DataFrame
    """
    working_df = df_indiv.loc[
        (df_indiv[col_time] >= min_time) & (
            df_indiv[col_indiv] == container)
    ]

    for row in working_df.iterrows():
        if row[1][col_val] > new_max:
            df_indiv.loc[
                (df_indiv[col_time] == row[1][col_time]) & (
                    df_indiv[col_indiv] == container), col_val
            ] = new_max


def order_csv_timestamp(path):
    """Order the CSV file by timestamp.

    :param path: initial folder path
    :type path: str
    """
    p_data = Path(path)
    data = pd.read_csv(
        p_data / 'container_usage.csv', index_col=False
    )
    print(data)
    data = data.sort_values(['timestamp'])
    print(data)
    data.to_csv(
        p_data / 'container_usage.csv', index=False)
    print('New CSV file written.')


def process_memory():
    """Get memory information."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_mb = mem_info.rss / 1024 / 1024
    return mem_usage_mb


def check_missing_entries_df(df):
    """Check if containers have missing entry in df."""
    # Find all unique timestamps and container IDs
    all_timestamps = df[it.tick_field].unique()
    all_containers = df[it.indiv_field].unique()

    # Create a complete MultiIndex of (timestamp, container_id)
    full_index = pd.MultiIndex.from_product([all_timestamps, all_containers], names=[it.tick_field, it.indiv_field])

    # Set index for the existing dataframe
    df.set_index([it.tick_field, it.indiv_field], inplace=True)

    # Reindex with the full index, filling missing rows
    df = df.reindex(full_index).reset_index()
    df[it.host_field] = df.groupby(it.indiv_field)[it.host_field].ffill()

    # Fill missing metrics values with 0.0
    for metric in it.metrics:
        df.fillna({metric: 0.0}, inplace=True)

    return df