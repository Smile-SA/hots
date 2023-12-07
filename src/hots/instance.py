"""Define the Instance class.

This class represents the problem we are facing, with its data, information, parameters.
This module provides also Instance-related methods.
"""

import math

from . import init as it
from . import node as nd


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

    def __init__(self, path, config):
        """Instance constructor.

        :param path: Filesystem path to the input files
        :type path: str
        :param config: Configuration dict from config file
        :type config: Dict
        """
        # TODO update by empty df_indiv and df_host => how to init df_host_meta ?
        (self.df_indiv,  # container usage
            self.df_host,  # node usage
            self.df_host_meta) = it.init_dfs(path)  # node meta deta

        # count of unique time values from timestamp column = 6
        self.time: int = self.df_indiv[it.tick_field].nunique()
        # count of unique machine_id values from machine_id column
        self.nb_nodes = self.df_host_meta[it.host_field].nunique()
        # count of unique container_ids from column container_id
        self.nb_containers = self.df_indiv[it.indiv_field].nunique()
        # gets default cluster numbers set to 3
        self.nb_clusters = config['clustering']['nb_clusters']

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

    def print_times(self, tick):
        """Print time informations.

        :param tick: _description_
        :type tick: int
        """
        print('Total time : ', self.time)
        print('Window duration : ', self.window_duration)
        print('Separation time : ', self.sep_time)
        print('Ticks : ', tick)

    def percentage_to_timestamp(self, config):
        """Transform percentage config time to timestamp.

        :param config: _description_
        :type config: Dict
        """
        # TODO consider 'tick' param as absolute, not percent ?
        self.window_duration = math.floor(
            self.time * int(config['analysis']['window_duration']) / 100
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
        # self.window_duration = 3
        if self.window_duration <= 1:
            self.window_duration += 1
        if self.sep_time <= 0:
            self.sep_time = 1
        if config['loop']['tick'] <= 0:
            config['loop']['tick'] = 1
        # if self.window_duration == config['loop']['tick']:
        #     self.window_duration += 1

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
