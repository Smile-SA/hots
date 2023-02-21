"""
============
hots.instance
============

Define the Instance class, which represents the problem we are facing, with its data, information,
parameters. Provide Instance-related methods.
"""

import math
from typing import Dict

import init as it
import node as nd


class Instance:
    """Description of a problem instance.

    Attributes:
        time: total time of dataset
        sep_time: time separating analysis and evaluation period
        window_duration: duration of time window for clustering
        nb_nodes: TODO: explain this data
        nb_containers: TODO: explain this data
        nb_clusters: TODO: explain this data
        df_indiv: TODO: explain this data
        df_host: TODO: explain this data
        df_host_meta: TODO: explain this data
        dict_id_n: TODO: explain this data
        dict_id_c: TODO: explain this data
    """

    def __init__(self, path: str, config: Dict):
        """Instance initialization

        Args:
            path: Filesystem path to the input files
            config: Configuration dict from config file
        """
        
        (self.df_indiv, # container usage 
         self.df_host, # node usage 
         self.df_host_meta) = it.init_dfs(path) # node meta deta 
        #new data
        self.df_container = self.df_indiv[self.df_indiv["timestamp"] < 2]
        self.df_container.reset_index()

        self.time: int = self.df_indiv[it.tick_field].nunique() # count of unique time values from timestamp column = 6
        self.nb_nodes = self.df_host_meta[it.host_field].nunique() # count of unique machine_id values from machine_id column
        self.nb_containers = self.df_indiv[it.indiv_field].nunique()  # count of unique container_ids from column container_id
        self.nb_clusters = config['clustering']['nb_clusters'] # gets default cluster numbers set to 3

        self.df_indiv = self.df_indiv.astype({
            it.indiv_field: str,
            it.host_field: str,
            it.tick_field: int})
        self.df_host = self.df_host.astype({
            it.host_field: str,
            it.tick_field: int})
        self.df_host_meta = self.df_host_meta.astype({it.host_field: str})

        self.df_host.sort_values(it.tick_field, inplace=True)
        self.df_indiv.sort_values(it.tick_field, inplace=True)
        self.df_host.set_index(
            [it.tick_field, it.host_field], inplace=True, drop=False)
        self.df_indiv.set_index(
            [it.tick_field, it.indiv_field], inplace=True, drop=False)

        self.percentage_to_timestamp(config)

        self.dict_id_n = nd.build_dict_id_nodes(self.df_host_meta)
        self.dict_id_c = {}
        self.kafkaConf = config['kafkaConf']

        self.print()

    # TODO rewrite with __str__
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

    def print_times(self, tick: int):
        """Print time informations."""
        print('Total time : ', self.time)
        print('Window duration : ', self.window_duration)
        print('Separation time : ', self.sep_time)
        print('Ticks : ', tick)

    def percentage_to_timestamp(self, config: Dict):
        """Transform percentage config time to timestamp."""
        # TODO consider 'tick' param as absolute, not percent ?
        self.window_duration = math.floor(
            self.time * int(config['analysis']['window_duration']) / 100 # window time 34
        )
        sep_nb_data = math.floor(
            self.time * int(config['analysis']['sep_time']) / 100  # seperation time 33
        )
        self.sep_time = self.df_indiv[it.tick_field].min() + sep_nb_data - 1
        if config['loop']['tick'] == 'default':
            config['loop']['tick'] = self.window_duration - 1
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

    def get_node_from_container(self, container_id: str) -> str:
        """Get node ID from container ID."""
        return (self.df_indiv.loc[
            self.df_indiv[it.indiv_field] == container_id
        ][it.host_field].to_numpy()[0])
