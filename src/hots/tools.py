"""Provide external tools usefull for the code."""

import os
import sys
from pathlib import Path

import pandas as pd

import psutil

from . import init as it


def print_size_vars(list_vars):
    """Display the size (in memory) of variables.

    :param list_vars: List of tuples containing variable names and their values
    :type list_vars: List[Tuple[str, Any]]
    """
    print('List of variables with sizes:')
    for name, size in sorted(
        ((name, sys.getsizeof(value)) for name, value in list_vars),
        key=lambda x: -x[1]
    )[:10]:
        print('{:>30}: {:>8}'.format(name, sizeof_fmt(size)))


def sizeof_fmt(num, suffix='B'):
    """Format a number as a human-readable memory size.

    :param num: The size in bytes
    :type num: float
    :param suffix: The suffix to append to the size (e.g., 'B' for bytes), defaults to 'B'
    :type suffix: str, optional
    :return: The formatted memory size as a string
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
    """Update a dataset by capping values at a maximum threshold.

    :param df_indiv: The input DataFrame containing individual data
    :type df_indiv: pd.DataFrame
    :param container: The container identifier to filter the data
    :type container: str
    :param col_indiv: The column name representing individual identifiers
    :type col_indiv: str
    :param new_max: The new maximum value to cap the data
    :type new_max: float
    :param col_val: The column name containing the values to be capped
    :type col_val: str
    :param col_time: The column name representing timestamps
    :type col_time: str
    :param min_time: The minimum timestamp to filter the data
    :type min_time: int
    :return: The updated DataFrame with capped values
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
    """Sort a CSV file by its timestamp column.

    :param path: Path to the folder containing the CSV file
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
    """Get the memory usage of the current process.

    :return: Memory usage in megabytes
    :rtype: float
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_mb = mem_info.rss / 1024 / 1024
    return mem_usage_mb


def check_missing_entries_df(df):
    """Check and fill missing entries in a DataFrame for all containers and timestamps.

    :param df: The input DataFrame containing container data
    :type df: pd.DataFrame
    :return: The updated DataFrame with missing entries filled
    :rtype: pd.DataFrame
    """
    # Find all unique timestamps and container IDs
    all_timestamps = df[it.tick_field].unique()
    all_containers = df[it.indiv_field].unique()

    # Create a complete MultiIndex of (timestamp, container_id)
    full_index = pd.MultiIndex.from_product(
        [all_timestamps, all_containers], names=[it.tick_field, it.indiv_field]
    )

    # Set index for the existing dataframe
    df.set_index([it.tick_field, it.indiv_field], inplace=True)

    # Reindex with the full index, filling missing rows
    df = df.reindex(full_index).reset_index()
    df[it.host_field] = df.groupby(it.indiv_field)[it.host_field].ffill()

    # Fill missing metrics values with 0.0
    for metric in it.metrics:
        df.fillna({metric: 0.0}, inplace=True)

    return df
