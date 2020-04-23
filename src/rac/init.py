"""
=========
rac init
=========

Provide stuff for initialization step (load DataFrames,
global variables)
"""

import typing as t
from pathlib import Path

import pandas as pd

# TODO list parameters here, or file to give in argument ?

# Global variables #

# We delete disk for the moment cause, not enough info
metrics = ['cpu', 'mem']

# Paths
data_dir = './data'

problem_name = 'generated_10'
problem_dir = f'{data_dir}/{problem_name}'
# instance_file = f'{data_dir}/{problem_name}/instance_file.txt'

input_file_container = f'{problem_dir}/container_usage.csv'
input_file_node = f'{problem_dir}/node_usage.csv'
input_file_node_meta = f'{problem_dir}/node_meta.csv'


# Functions definitions #


def build_df_from_containers(df_containers: pd.DataFrame) -> pd.DataFrame:
    """Build the `df_nodes` from containers df."""
    df_nodes = pd.DataFrame(data=None)
    for time in df_containers['timestamp'].unique():
        for node in df_containers['machine_id'].unique():
            df_nodes = df_nodes.append(
                {
                    'timestamp': time,
                    'machine_id': node,
                    'cpu': df_containers.loc[
                        (df_containers['timestamp'] == time) & (
                            df_containers['machine_id'] == node)
                    ]['cpu'].sum(),
                    'mem': df_containers.loc[
                        (df_containers['timestamp'] == time) & (
                            df_containers['machine_id'] == node)
                    ]['mem'].sum()
                }, ignore_index=True
            )
    return df_nodes


def df_from_csv(file: Path) -> pd.DataFrame:
    """Load DataFrame from CSV file."""
    return pd.read_csv(
        file, index_col=False)


# TODO check if files exist ?
# TODO optionnal node file ?
def init_dfs(data: str) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform CSV files reading in data folder."""
    p_data = Path(data)

    if Path(p_data / 'node_usage.csv').is_file():
        return (df_from_csv(p_data / 'container_usage.csv'),
                df_from_csv(p_data / 'node_usage.csv'),
                df_from_csv(p_data / 'node_meta.csv'))
    else:
        print('We need to build node usage from containers ...')
        df_containers = df_from_csv(p_data / 'container_usage.csv')
        df_nodes = build_df_from_containers(df_containers)
        df_nodes.to_csv(p_data / 'node_usage.csv', index=False)
        return (df_containers,
                df_nodes,
                df_from_csv(p_data / 'node_meta.csv'))
