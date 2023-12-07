"""Provide stuff for initialization step (load DataFrames, global variables)."""

import json
from pathlib import Path

import pandas as pd

from . import reader

# Functions definitions #


def build_df_from_containers(df_indiv):
    """Build the `df_host` from containers df.

    :param df_indiv: _description_
    :type df_indiv: pd.DataFrame
    :return: _description_
    :rtype: pd.DataFrame
    """
    dict_agg = {}
    for metric in metrics:
        dict_agg[metric] = 'sum'

    df_host = df_indiv.groupby(
        [tick_field, host_field], as_index=False).agg(dict_agg)

    return df_host


def df_from_csv(file):
    """Load DataFrame from CSV file.

    :param file: _description_
    :type file: Path
    :return: _description_
    :rtype: pd.DataFrame
    """
    return pd.read_csv(
        file, index_col=False)


# TODO check if files exist ?
def init_dfs(data):
    """Perform CSV files reading in data folder.

    :param data: _description_
    :type data: str
    :return: _description_
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
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


def read_params(
    path, k, tau, method, cluster_method, param, output_path, kafka_var
):
    """Get parameters from file and build the Dict config object.

    :param path: _description_
    :type path: str
    :param k: _description_
    :type k: int
    :param tau: _description_
    :type tau: int
    :param method: _description_
    :type method: str
    :param cluster_method: _description_
    :type cluster_method: str
    :param param: _description_
    :type param: str
    :param output_path: _description_
    :type output_path: str
    :param kafka_var: streaming platform
    :type kafka_var: bool
    :raises ValueError: _description_
    :raises ValueError: _description_
    :return: _description_
    :rtype: Dict
    """
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
        output_path = Path(
            '%sk%d_tau%d_%s_%s' % (
                path,
                config['clustering']['nb_clusters'],
                int(tau),
                method, cluster_method
            ))
    output_path.mkdir(parents=True, exist_ok=True)
    define_globals(output_path, config, kafka_var)
    if method not in methods:
        raise ValueError('Method %s is not accepted' % method)
    if cluster_method not in cluster_methods:
        raise ValueError('Updating clustering method %s is not accepted' % cluster_method)
    return (config, str(output_path))


def set_loop_results():
    """Create the dataframe for loop results.

    :return: _description_
    :rtype: pd.DataFrame
    """
    return pd.DataFrame(columns=[
        'num_loop', 'init_silhouette', 'init_delta',
        'clust_conf_nodes', 'clust_conf_edges',
        'clust_max_deg', 'clust_mean_deg',
        'clust_changes',
        'place_conf_nodes', 'place_conf_edges',
        'place_max_deg', 'place_mean_deg',
        'place_changes',
        'end_silhouette', 'end_delta', 'loop_time'
    ])


def set_times_df():
    """Create the dataframe for times info.

    :return: _description_
    :rtype: pd.DataFrame
    """
    return pd.DataFrame(columns=[
        'num_loop', 'action', 'time'
    ])


def define_globals(p_path, config, kafka_var):
    """Define the fields, as global variables, from config.

    :param p_path: _description_
    :type p_path: Path
    :param config: _description_
    :type config: Dict
    :param kafka_var: streaming platform
    :type kafka_var: bool
    """
    global indiv_field
    global host_field
    global tick_field
    global metrics

    global methods
    global cluster_methods

    global global_results
    global loop_results
    global times_df
    # global node_results

    global results_file
    global additional_results_file
    global optim_file
    global clustering_file

    global dict_agg_metrics

    global renderer

    global streamkm_model

    global csv_reader

    global avro_deserializer
    global s_entry
    global kafka_producer
    global kafka_consumer
    global kafka_topics
    global kafka_schema
    global kafka_schema_url
    global tick_time
    global time_at
    global memory_usage
    global total_mem_use

    indiv_field = config['data']['individual_field']
    host_field = config['data']['host_field']
    tick_field = config['data']['tick_field']
    metrics = config['data']['metrics']

    methods = ['init', 'spread', 'iter-consol', 'heur', 'loop']
    cluster_methods = ['loop-cluster',
                       'kmeans-scratch',
                       'stream-km']

    loop_results = set_loop_results()
    times_df = set_times_df()

    results_file = open(p_path / 'results.log', 'w')
    # additional_results_file = open(p_path / 'results.log', 'w')
    optim_file = open(p_path / 'optim_logs.log', 'w')
    clustering_file = open(p_path / 'clustering_logs.log', 'w')

    s_entry = True

    if kafka_var:
        kafka_topics = config['kafkaConf']['topics']
        reader.kafka_availability(config)
        kafka_producer = reader.get_producer(config)
        kafka_consumer = reader.get_consumer(config)
        kafka_schema = config['kafkaConf']['schema']
        kafka_schema_url = config['kafkaConf']['schema_url']
    dict_agg_metrics = {}
    for metric in metrics:
        dict_agg_metrics[metric] = 'sum'

    renderer = config['plot']['renderer']
