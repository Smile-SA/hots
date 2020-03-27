# coding=utf-8

# print(__doc__)

from pathlib import Path
import pandas as pd

# TODO list parameters here, or file to give in argument ?

### Global variables ###

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


### Functions definitions ###

def df_from_csv(filename):
    return pd.read_csv(
        filename, index_col=False)


def init_dfs():
    return (df_from_csv(input_file_container), df_from_csv(input_file_node))


def init_dfs_meta():
    return df_from_csv(input_file_node_meta)
