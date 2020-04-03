"""
=========
rac init
=========

Provide stuff for initialization step (load DataFrames,
global variables)
"""

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


def df_from_csv(filename: str) -> pd.DataFrame:
    """Load DataFrame from CSV file."""
    return pd.read_csv(
        filename, index_col=False)


# TODO check if files exist ?
# TODO optionnal node file ?
def init_dfs(data: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Perform CSV files reading in data folder."""
    return (df_from_csv(f'{data}/container_usage.csv'),
            df_from_csv(f'{data}/node_usage.csv'),
            df_from_csv(f'{data}/node_meta.csv'))
