"""
=========
cots allocation
=========

Provide resource allocation related functions to handle this problem.
"""

from typing import Dict

import pandas as pd

from . import init as it
from .instance import Instance


def check_constraints(my_instance: Instance, config: Dict) -> bool:
    """Check if allocation constraints are satisfied or not."""
    satisfied = True
    print(config)
    print(my_instance.df_host)
    if my_instance.df_host[it.host_field].nunique() > config['objective']['open_nodes']:
        print('Too many open nodes !')
        satisfied = False
    elif my_instance.df_host[it.host_field].nunique() < config['objective']['open_nodes']:
        print('Less open nodes than the objective !')
        satisfied = False
    else:
        print('Right number of open nodes.')

    if max(my_instance.df_host[it.metrics[0]]) > (
        config['objective']['target_load_CPU'] * (
            my_instance.df_host_meta[it.metrics[0]].to_numpy()[0])
    ):
        print('Max resources used > wanted max !')
        satisfied = False
        change_max_alloc(my_instance, config)
    else:
        print('Max used resources ok.')

    return satisfied


# TODO define max by container
# TODO change alloc only for containers in node > max
def change_max_alloc(my_instance: Instance, config: Dict):
    max_ok = False
    max_goal = config['objective']['target_load_CPU'] * (
        my_instance.df_host_meta[it.metrics[0]].to_numpy()[0]
    )
    current_max_by_node = get_max_by_node(my_instance.df_indiv)
    while not max_ok:
        decrease = 0.1
        for container in my_instance.df_indiv[it.indiv_field]:
            node = my_instance.df_indiv.loc[my_instance.df_indiv[
                it.indiv_field] == container][it.host_field].to_numpy()[0]
            current_max_by_node[node][container] = current_max_by_node[
                node][container] - (current_max_by_node[
                    node][container] * decrease)
            if is_max_goal_ok(current_max_by_node, max_goal):
                max_ok = True
                break
    print(current_max_by_node)
    input()


def get_max_by_node(df_indiv: pd.DataFrame) -> Dict:
    max_by_node = {}
    for node, node_data in df_indiv.groupby(it.host_field):
        node_max = {}
        for container, cont_data in node_data.groupby(node_data[it.indiv_field]):
            node_max[container] = max(cont_data[it.metrics[0]])
        max_by_node[node] = node_max
    return max_by_node


def is_max_goal_ok(current_max_by_node: Dict, max_goal: float) -> bool:
    for node in current_max_by_node.keys():
        if sum(current_max_by_node[node].values()) > max_goal:
            print('Not ok')
            return False
    return True
