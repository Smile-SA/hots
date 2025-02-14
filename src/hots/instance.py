"""Define the Instance class.

This class represents the problem we are facing,
with its data, information, parameters.
This module provides also Instance-related methods.
"""

import math
import os

from dotenv import load_dotenv

import pandas as pd

try:
    import requests
except ImportError:
    _has_kafka = False
else:
    _has_kafka = True

from . import init as it
from . import node as nd

load_dotenv()
connector_url = os.getenv('CONNECTOR_URL')


class Instance:
    """Description of a problem instance.

    :param df_indiv: _description_
    :type df_indiv: pd.DataFrame
    :param df_host: _description_
    :type df_host: pd.DataFrame
    :param df_host_meta: _description_
    :type df_host_meta: pd.DataFrame
    :param time: _description_
    :type time: int
    :param nb_nodes: _description_
    :type nb_nodes: int
    :param nb_containers: _description_
    :type nb_containers: int
    :param nb_clusters: _description_
    :type nb_clusters: int
    :param dict_id_n: _description_
    :type dict_id_n: Dict
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    """

    def __init__(self, config):
        """Instance constructor.

        :param config: Configuration dict from config file
        :type config: Dict
        """
        self.nb_clusters = config['clustering']['nb_clusters']

        # self.df_indiv = self.df_indiv.astype({
        #     it.indiv_field: str,
        #     it.host_field: str,
        #     it.tick_field: int})
        # self.df_host = self.df_host.astype({
        #     it.host_field: str,
        #     it.tick_field: int})
        # self.df_host_meta = self.df_host_meta.astype({it.host_field: str})

        # self.df_host.sort_values(it.tick_field, inplace=True)
        # self.df_indiv.sort_values(it.tick_field, inplace=True)
        # self.df_host.set_index(
        #     [it.tick_field, it.host_field], inplace=True, drop=False)
        # self.df_indiv.set_index(
        #     [it.tick_field, it.indiv_field], inplace=True, drop=False)

        # TODO remove as we are now in streaming => direct timestamp given
        # self.percentage_to_timestamp(config)

        self.sep_time = int(config['analysis']['sep_time'])
        self.window_duration = int(config['loop']['window_duration'])
        self.tick = int(config['loop']['tick'])

        # self.dict_id_n = nd.build_dict_id_nodes(self.df_host_meta)
        self.dict_id_c = {}
        self.container_to_id = {}
        self.last_assigned_id = 0

    # TODO rewrite with __str__
    # TODO not relevant with streaming
    def print(self):
        """Print Instance information."""
        it.results_file.writelines(['### Problem instance informations ###\n',
                                    'Time considered : %d\n' % self.time,
                                    '%d nodes -- ' % self.nb_nodes,
                                    '%d containers\n' % self.nb_containers,
                                    '\n### Parameters ###\n'
                                    'clusters : %d\n' % self.nb_clusters,
                                    'tau : %d (%f%%)\n' % (
                                        self.window_duration,
                                        (self.window_duration / self.time)),
                                    '\n'])

    def print_times(self):
        """Print time informations.

        :param tick: _description_
        :type tick: int
        """
        # print('Total time : ', self.time)
        print('Window duration : ', self.window_duration)
        print('Separation time : ', self.sep_time)
        print('Ticks : ', self.tick)

    def percentage_to_timestamp(self, config):
        """Transform percentage config time to timestamp.

        :param config: _description_
        :type config: Dict
        """
        # TODO consider 'tick' param as absolute, not percent ?
        self.window_duration = math.floor(
            self.time * int(config['loop']['window_duration']) / 100
        )
        sep_nb_data = math.floor(
            self.time * int(config['analysis']['sep_time']) / 100
        )
        self.sep_time = self.df_indiv[it.tick_field].min() + sep_nb_data - 1
        if config['loop']['tick'] == 'default':
            config['loop']['tick'] = self.window_duration - 1
            # config['loop']['tick'] = 2
        else:
            config['loop']['tick'] = math.floor(
                self.time * int(config['loop']['tick']) / 100
            ) - 1
        if self.window_duration <= 1:
            self.window_duration += 1
        if self.sep_time <= 0:
            self.sep_time = 1
        if config['loop']['tick'] <= 0:
            config['loop']['tick'] = 1
        # if self.window_duration == config['loop']['tick']:
        #     self.window_duration += 1

    def set_meta_info(self):
        """Define the number of containers and nodes from data."""
        self.nb_nodes = self.df_host_meta[it.host_field].nunique()
        self.nb_containers = self.df_indiv[it.indiv_field].nunique()

    def set_host_meta(self, host_meta_path):
        """Create the dataframe for host meta data from first data."""
        self.df_host_meta = it.df_from_csv(host_meta_path)
        self.dict_id_n = nd.build_dict_id_nodes(self.df_host_meta)
        self.nb_nodes = self.df_host_meta[it.host_field].nunique()

    def init_host_evo(self):
        """Initialize the data evolution in hosts DataFrame."""
        self.df_host_evo = pd.DataFrame(columns=self.df_host.columns)

    def get_or_create_container_id(self, container_id):
        """Get app ID or create it in global matching dict.

        :param container_id: ID from external data
        :type container_id: str
        :return: numeric app ID
        :rtype: int
        """
        if container_id not in self.container_to_id:
            self.dict_id_c[self.last_assigned_id] = container_id
            self.container_to_id[container_id] = self.last_assigned_id
            self.last_assigned_id += 1
        return self.container_to_id[container_id]

    def get_node_from_container(self, container_id):
        """Get node ID from container ID.

        :param container_id: _description_
        :type container_id: str
        :return: _description_
        :rtype: str
        """
        return (self.df_indiv.loc[
            self.df_indiv[it.indiv_field] == container_id
        ][it.host_field].to_numpy()[0])

    def get_node_information(self):
        """Get node information from environment.

        :return: Node information
        :rtype: Dict
        """
        url = f'{connector_url}/vm/data'
        response = requests.get(url)
        node_data = {}
        if response.status_code == 200:
            node_data = response.json()
            self.df_host_meta = pd.DataFrame(node_data)
            self.dict_id_n = nd.build_dict_id_nodes(self.df_host_meta)
            self.nb_nodes = self.df_host_meta[it.host_field].nunique()
        else:
            # If the request was not successful, print the error status code
            print(f'Error: {response.status_code}')
        return node_data

    def start_stream():
        """Start stream data in environment."""
        url = f'{connector_url}/start_stream'
        response = requests.get(url)
        if response.status_code == 200:
            print(response)
        else:
            # If the request was not successful, print the error status code
            print(f'Error: {response.status_code}')

    def stop_stream():
        """Stop stream data in environment."""
        url = f'{connector_url}/stop_stream'
        response = requests.get(url)
        if response.status_code == 200:
            print(response)
        else:
            # If the request was not successful, print the error status code
            print(f'Error: {response.status_code}')

    def clear_kafka_topics():
        """Clear Kafka topics in environment."""
        url = f'{connector_url}/clear_topics'
        response = requests.get(url)
        if response.status_code == 200:
            print(response)
        else:
            # If the request was not successful, print the error status code
            print(f'Error: {response.status_code}')
