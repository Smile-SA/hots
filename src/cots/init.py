"""
=========
cots init
=========

Provide stuff for initialization step (load DataFrames,
global variables)
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Functions definitions #


def build_df_from_containers(df_indiv: pd.DataFrame) -> pd.DataFrame:
    """Build the `df_host` from containers df."""
    df_host = pd.DataFrame(data=None)
    for time in df_indiv[tick_field].unique():
        for node in df_indiv[host_field].unique():
            temp_df = pd.Series({
                tick_field: time,
                host_field: node
            })
            for metric in metrics:
                metric_val = df_indiv.loc[
                    (df_indiv[tick_field] == time) & (
                        df_indiv[host_field] == node)
                ][metric].sum()
                temp_df[metric] = metric_val
            df_host = df_host.append(
                temp_df, ignore_index=True
            )
    return df_host


def df_from_csv(file: Path) -> pd.DataFrame:
    """Load DataFrame from CSV file."""
    return pd.read_csv(
        file, index_col=False)


# TODO check if files exist ?
# TODO optionnal node file ?
def init_dfs(data: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform CSV files reading in data folder."""
    p_data = Path(data)

    if Path(p_data / 'node_usage.csv').is_file():
        return (df_from_csv(p_data / 'container_usage.csv'),
                df_from_csv(p_data / 'node_usage.csv'),
                df_from_csv(p_data / 'node_meta.csv'))
    else:
        print('We need to build node usage from containers ...')
        df_indiv = df_from_csv(p_data / 'container_usage.csv')
        df_host = build_df_from_containers(df_indiv)
        df_host.to_csv(p_data / 'node_usage.csv', index=False)
        return (df_indiv,
                df_host,
                df_from_csv(p_data / 'node_meta.csv'))


def read_params(params_file: str) -> Dict:
    """Get parameters from file and build the Dict config object."""
    with open(params_file, 'r') as f:
        config = json.load(f)
    define_globals(config)
    return config


def define_globals(config: Dict):
    """Define the fields, as global variables, from config."""
    global indiv_field
    global host_field
    global tick_field
    global metrics

    global renderer

    indiv_field = config['data']['individual_field']
    host_field = config['data']['host_field']
    tick_field = config['data']['tick_field']
    metrics = config['data']['metrics']

    renderer = config['plot']['renderer']
