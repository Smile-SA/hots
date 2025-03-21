"""Provide stuff for initialization step."""

import json
from pathlib import Path

import pandas as pd

from . import reader
from .instance import Instance

# Functions definitions #


def build_df_from_containers(df_indiv):
    """Build the `df_host` from containers df.

    :param df_indiv: Individual consumption data
    :type df_indiv: pd.DataFrame
    :return: Node consumption
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

    :param file: Data file path
    :type file: Path
    :return: Dataframe with data
    :rtype: pd.DataFrame
    """
    return pd.read_csv(
        file, index_col=False)


# TODO check if files exist ?
def init_dfs(data):
    """Perform CSV files reading in data folder.

    :param data: Data folder path
    :type data: str
    :return: Individual consumption, Node consumption, Nodes capacity
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
    path, k, tau, method, cluster_method, output_path, kafka_var
):
    """Get parameters from file and build the Dict config object.

    :param path: Config file path
    :type path: str
    :param k: Number of clusters
    :type k: int
    :param tau: Time window size
    :type tau: int
    :param method: Method for global process
    :type method: str
    :param cluster_method: Clustering method
    :type cluster_method: str
    :param output_path: Path for output files
    :type output_path: str
    :param kafka_var: streaming platform
    :type kafka_var: bool
    :return: Configs object, Path for output, Parent folder path
    :rtype: Dict
    """
    p_path = Path(path)
    # TODO check if path is json file
    with open(p_path, 'r') as f:
        config = json.load(f)
    if k is not None:
        config['clustering']['nb_clusters'] = k
    if tau is not None:
        config['loop']['window_duration'] = tau
        config['loop']['tick'] = tau
    else:
        tau = config['loop']['window_duration']
    if output_path is not None:
        output_path = Path(output_path)
    else:
        output_path = Path(
            '%s/k%d_tau%d_%s_%s' % (
                str(p_path.parent),
                config['clustering']['nb_clusters'],
                int(tau),
                method, cluster_method
            ))
    output_path.mkdir(parents=True, exist_ok=True)
    define_globals(output_path, config, kafka_var)
    if method not in methods:
        raise ValueError('Method %s is not accepted' % method)
    if cluster_method not in cluster_methods:
        raise ValueError(
            'Updating clustering method %s is not accepted' % cluster_method
        )
    return (config, str(output_path), str(p_path.parent))


def set_loop_results():
    """Create the dataframe for loop results.

    :return: Loop results dataframe
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

    :return: Times results dataframe
    :rtype: pd.DataFrame
    """
    return pd.DataFrame(columns=[
        'num_loop', 'action', 'time'
    ])


def define_globals(p_path, config, kafka_var):
    """Define the fields, as global variables, from config.

    :param p_path: Parent folder path
    :type p_path: Path
    :param config: Configs Dict
    :type config: Dict
    :param kafka_var: streaming platform
    :type kafka_var: bool
    """
    global my_instance

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

    global streamkm_model

    global csv_file
    global csv_reader
    global csv_queue

    global use_kafka

    global avro_deserializer
    global s_entry
    global end
    global kafka_producer
    global kafka_consumer
    global kafka_topics
    global kafka_schema
    global kafka_schema_url
    global connector_url
    global tick_time
    global time_at
    global memory_usage
    global total_mem_use

    global pending_changes

    indiv_field = config['data']['individual_field']
    host_field = config['data']['host_field']
    tick_field = config['data']['tick_field']
    metrics = config['data']['metrics']

    methods = ['init', 'spread', 'iter-consol', 'heur', 'loop']
    cluster_methods = ['loop-cluster',
                       'kmeans-scratch',
                       'stream-km']

    # TODO possibility to disable results compute / building
    loop_results = set_loop_results()
    times_df = set_times_df()

    results_file = open(p_path / 'results.log', 'w')
    # additional_results_file = open(p_path / 'results.log', 'w')
    optim_file = open(p_path / 'optim_logs.log', 'w')
    clustering_file = open(p_path / 'clustering_logs.log', 'w')

    s_entry = True
    end = False

    use_kafka = kafka_var
    if kafka_var:
        kafka_topics = config['kafkaConf']['topics']
        reader.kafka_availability(config)
        Instance.clear_kafka_topics()
        kafka_producer = reader.get_producer(config)
        kafka_consumer = reader.get_consumer(config)
        kafka_schema = config['kafkaConf']['schema']
        kafka_schema_url = config['kafkaConf']['schema_url']
        connector_url = config['kafkaConf']['connector_url']
    dict_agg_metrics = {}
    for metric in metrics:
        dict_agg_metrics[metric] = 'sum'

    pending_changes = {}
