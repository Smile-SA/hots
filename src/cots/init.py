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


def build_df_from_containers_old(df_indiv: pd.DataFrame) -> pd.DataFrame:
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


def build_df_from_containers(df_indiv: pd.DataFrame) -> pd.DataFrame:
    """Build the `df_host` from containers df."""
    dict_agg = {}
    for metric in metrics:
        dict_agg[metric] = 'sum'

    df_host = df_indiv.groupby(
        [tick_field, host_field], as_index=False).agg(dict_agg)

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


# TODO generalize
def init_algo_dfs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Initialize host dataframes for specific methods."""
    spread_df = pd.DataFrame()
    heur_df = pd.DataFrame()
    loop_df = pd.DataFrame()

    return (spread_df, heur_df, loop_df)


def read_params(path: str, k: int, tau: int) -> Dict:
    """Get parameters from file and build the Dict config object."""
    p_path = Path(path)
    with open(p_path / 'params.json', 'r') as f:
        config = json.load(f)
    if k is not None:
        config['clustering']['nb_clusters'] = k
    if tau is not None:
        config['analysis']['window_duration'] = tau + 1
        config['analysis']['eval_time'] = tau
        config['loop']['tick'] = tau
    output_path = Path(path + 'k' + str(
        config['clustering']['nb_clusters']) + '_' + 'tau' + str(
            config['loop']['tick']))
    output_path.mkdir(parents=True, exist_ok=True)
    define_globals(output_path, config)
    return (config, str(output_path))


def set_loop_results() -> pd.DataFrame:
    """Create the dataframe for loop results."""
    return pd.DataFrame(columns=[
        'num_loop', 'clust_changes', 'place_changes', 'loop_time'
    ])


def define_globals(p_path: Path, config: Dict):
    """Define the fields, as global variables, from config."""
    global indiv_field
    global host_field
    global tick_field
    global metrics

    global main_results
    global loop_results
    # global node_results

    global results_file
    global main_results_file
    global additional_results_file

    global renderer

    indiv_field = config['data']['individual_field']
    host_field = config['data']['host_field']
    tick_field = config['data']['tick_field']
    metrics = config['data']['metrics']

    main_results = []
    loop_results = set_loop_results()

    results_file = open(p_path / 'results.log', 'w')
    main_results_file = open(p_path / 'main_results.csv', 'w')
    # additional_results_file = open(p_path / 'results.log', 'w')

    renderer = config['plot']['renderer']
