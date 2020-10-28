"""
============
cots.instance
============

Define the Instance class, which represents the problem we are facing, with its data, information,
parameters. Provide Instance-related methods.
"""

import math
from typing import Dict

from . import init as it
from . import node as nd


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

    def __init__(self, data: str, config: Dict):
        """Instance initialization

        Args:
            data: Filesystem path to the input files
            nb_clusters: WARNING: seems useless !
        """
        (self.df_indiv,
         self.df_host,
         self.df_host_meta) = it.init_dfs(data)

        # self.indiv_field = config['data']['individual_field']
        # self.host_field = config['data']['host_field']
        # self.tick_field = config['data']['tick_field']
        # self.metrics = config['data']['metrics']
        print(it.indiv_field)

        self.time: int = self.df_indiv[it.tick_field].nunique()
        if config['analysis']['window_duration'] == 'default':
            self.sep_time: int = math.floor(self.time / 2) + self.df_indiv[
                it.tick_field].min()
            self.window_duration = self.df_indiv.loc[
                self.df_indiv[it.tick_field] <= self.sep_time
            ][it.tick_field].nunique()
        else:
            self.window_duration = config['analysis']['window_duration']
            self.sep_time: int = self.df_indiv[
                it.tick_field].min() + self.window_duration - 1

        self.nb_nodes = self.df_host_meta[it.host_field].nunique()
        self.nb_containers = self.df_indiv[it.indiv_field].nunique()
        self.nb_clusters = config['clustering']['nb_clusters']

        self.df_host.sort_values(it.tick_field, inplace=True)
        self.df_indiv.sort_values(it.tick_field, inplace=True)
        self.df_host.set_index(
            [it.tick_field, it.host_field], inplace=True, drop=False)
        self.df_indiv.set_index(
            [it.tick_field, it.indiv_field], inplace=True, drop=False)

        self.dict_id_n = nd.build_dict_id_nodes(self.df_host_meta)
        self.dict_id_c = {}

        self.print()

    # TODO rewrite with __str__
    def print(self):
        """Print Instance information."""
        print('\n')
        print('### Problem instance informations ###')
        print('Time considered : %d' % self.time)
        print('%d nodes' % self.nb_nodes)
        print('%d containers' % self.nb_containers)
        print('\n')
        # Not useful ?
        # print('%d clusters' % self.nb_clusters)

    # TODO rewrite with only one f.write

    def instance_in_file_before(self, filename: str):
        """Write Instance information in file."""
        f = open(filename, 'w')
        f.write('### Problem instance informations ###\n')
        f.write('Time considered : %d\n' % self.time)
        f.write('%d nodes -- ' % self.nb_nodes)
        f.write('%d containers\n' % self.nb_containers)
        f.write('\n')

        f.write('### Variance before optimization ###\n')
        var, global_var = nd.get_nodes_variance(self.df_host, self.time, 2)
        f.write(str(var))
        f.write('\nGlobal variance : %s\n' % str(global_var))

        f.write('\n### Nodes state at the beginning ###\n')

        # for node in self.nodes_:
        #     node.print_inFile(f, 1)
        #     f.write('\n')

        f.write('\n')
        f.close()

    def instance_in_file_after(self, filename: str):
        """Write Instance information after optimization step."""
        f = open(filename, 'a')

        f.write('\n### Variance after optimization ###\n')
        var, global_var = nd.get_nodes_variance(self.df_host, self.time, 2)
        f.write(str(var))
        f.write('\nGlobal variance : %s\n' % str(global_var))

        f.close()

    def get_node_from_container(self, container_id: str) -> str:
        """Get node ID from container ID."""
        return (self.df_indiv.loc[
            self.df_indiv[it.indiv_field] == container_id
        ][it.host_field].to_numpy()[0])
