"""
=========
rac instance
=========

Define the Instance class, which represents the problem we are facing,
with its data, information, parameters. Provide Instance-related methods.
"""

import math

# print(__doc__)
from . import node as nd
from . import container
from .init import init_dfs, init_dfs_meta
import math


class Instance:
    """Class describing a instance with following attributes :
                - time : total time of dataset
                - sep_time : time separating analysis and evaluation period
                - window_duration : duration of time window for clustering

                - nb_nodes
                - nb_containers
                - nb_clusters

                - df_containers
                - df_nodes
                - df_nodes_meta

                - dict_id_n
                - dict_id_c
    """

    # functions #
    def __init__(self, data: str, nb_clusters: int = 3):

        (self.df_containers,
         self.df_nodes,
         self.df_nodes_meta) = init_dfs(data)

        (self.df_containers, self.df_nodes) = init_dfs()
        self.df_nodes_meta = init_dfs_meta()
        self.time = self.df_containers['timestamp'].nunique()
        self.sep_time = math.ceil(self.time / 2)
        self.window_duration = self.df_containers.loc[
            self.df_containers['timestamp'] <= self.sep_time
        ]['timestamp'].nunique()

        self.nb_nodes = self.df_nodes['machine_id'].nunique()
        self.nb_containers = self.df_containers['container_id'].nunique()
        self.nb_clusters = nb_clusters

        self.df_nodes.sort_values('timestamp', inplace=True)
        self.df_containers.sort_values('timestamp', inplace=True)
        self.df_nodes.set_index(
            ['timestamp', 'machine_id'], inplace=True, drop=False)
        self.df_containers.set_index(
            ['timestamp', 'container_id'], inplace=True, drop=False)

        self.dict_id_n = nd.build_dict_id_nodes(self.df_nodes)
        self.dict_id_c = container.build_dict_id_containers(self.df_containers)

        self.print()

    # TODO rewrite with __str__
    def print(self):
        print('\n')
        print('### Problem instance informations ###')
        print('Time considered : %d' % self.time)
        print('%d nodes' % self.nb_nodes)
        print('%d containers' % self.nb_containers)
        print('\n')
        # Not useful ?
        # print('%d clusters' % self.nb_clusters)

    # TODO rewrite with only one f.write

    def instance_inFile_before(self, filename: str):
        f = open(filename, 'w')
        f.write('### Problem instance informations ###\n')
        f.write('Time considered : %d\n' % self.time)
        f.write('%d nodes -- ' % self.nb_nodes)
        f.write('%d containers\n' % self.nb_containers)
        f.write('\n')

        f.write('### Variance before optimization ###\n')
        var, global_var = nd.get_nodes_variance(self.df_nodes, self.time, 2)
        f.write(str(var))
        f.write('\nGlobal variance : %s\n' % str(global_var))

        f.write('\n### Nodes state at the beginning ###\n')

        # for node in self.nodes_:
        #     node.print_inFile(f, 1)
        #     f.write('\n')

        f.write('\n')
        f.close()

    def instance_inFile_after(self, filename: str):
        f = open(filename, 'a')

        f.write('\n### Variance after optimization ###\n')
        var, global_var = nd.get_nodes_variance(self.df_nodes, self.time, 2)
        f.write(str(var))
        f.write('\nGlobal variance : %s\n' % str(global_var))

        f.close()

    def get_node_from_container(self, container_id: str) -> str:
        return (self.df_containers.loc[
            self.df_containers['container_id'] == container_id
        ]['machine_id'].to_numpy()[0])
