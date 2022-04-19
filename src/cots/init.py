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


def read_params(path: str, k: int, tau: int, method: str,
                param: str, output_path: str) -> Dict:
    """Get parameters from file and build the Dict config object."""
    p_path = Path(path)
    if param is not None:
        config_path = Path(param)
    elif Path(p_path / 'params.json').exists():
        config_path = p_path / 'params.json'
    else:
        config_path = 'tests/params_default.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    if k is not None:
        config['clustering']['nb_clusters'] = k
    if tau is not None:
        config['analysis']['window_duration'] = tau
        config['loop']['tick'] = tau
    else:
        tau = config['analysis']['window_duration']
    if output_path is not None:
        output_path = Path(output_path)
    else:
        output_path = Path(path + 'k' + str(
            config['clustering']['nb_clusters']) + '_' + 'tau' + str(
            tau) + '_' + method)
    output_path.mkdir(parents=True, exist_ok=True)
    define_globals(output_path, config)
    if method not in methods:
        raise ValueError('Method %s is not accepted' % method)
    return (config, str(output_path))


def set_loop_results() -> pd.DataFrame:
    """Create the dataframe for loop results."""
    return pd.DataFrame(columns=[
        'num_loop', 'init_delta',
        'clust_conf_nodes', 'clust_conf_edges', 'clust_changes',
        'place_conf_nodes', 'place_conf_edges', 'place_changes',
        'end_delta', 'loop_time'
    ])


def define_globals(p_path: Path, config: Dict):
    """Define the fields, as global variables, from config."""
    global indiv_field
    global host_field
    global tick_field
    global metrics

    global methods

    global main_results
    global loop_results
    # global node_results

    global results_file
    global main_results_file
    global additional_results_file
    global optim_file
    global clustering_file

    global dict_agg_metrics

    global renderer

    indiv_field = config['data']['individual_field']
    host_field = config['data']['host_field']
    tick_field = config['data']['tick_field']
    metrics = config['data']['metrics']

    methods = ['init', 'spread', 'iter-consol', 'heur', 'loop', 'loop_v2', 'loop_kmeans']

    main_results = []
    loop_results = set_loop_results()

    results_file = open(p_path / 'results.log', 'w')
    main_results_file = open(p_path / 'main_results.csv', 'w')
    # additional_results_file = open(p_path / 'results.log', 'w')
    optim_file = open(p_path / 'optim_logs.log', 'w')
    clustering_file = open(p_path / 'clustering_logs.log', 'w')

    dict_agg_metrics = {}
    for metric in metrics:
        dict_agg_metrics[metric] = 'sum'

    renderer = config['plot']['renderer']
