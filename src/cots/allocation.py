"""
=========
cots allocation
=========

Provide resource allocation related functions to handle this problem.
"""

import math
from typing import Dict

import numpy as np

import pandas as pd

from . import init as it
from .instance import Instance
from .placement import spread_containers
from .tools import change_max_dataset


def check_constraints(my_instance: Instance,
                      working_df_indiv: pd.DataFrame, config: Dict) -> bool:
    """Check if allocation constraints are satisfied or not."""
    satisfied = False
    print(config)
    print(my_instance.df_host)

    # while not satisfied:
    current_max_by_node = get_max_by_node(my_instance.df_indiv)
    if get_total_max(current_max_by_node) > get_abs_goal_load(my_instance, config):
        print('Max resources used > wanted max ! (new)')
        satisfied = False
        change_max_bound(my_instance, config, working_df_indiv[it.tick_field].max())
    else:
        satisfied = True

    if max(my_instance.df_host[it.metrics[0]]) > get_abs_goal_load(my_instance, config):
        print('Max resources used > wanted max !')
        satisfied = False
        change_max_bound(my_instance, config, working_df_indiv[it.tick_field].max())
    else:
        print('Max used resources ok.')
        satisfied = True

    if my_instance.df_host[it.host_field].nunique() > config['objective']['open_nodes']:
        print('Too many open nodes !')
        satisfied = False
        # TODO Test if we can do it
        # TODO Do not test with previous data, as it's not supposed to change
        # move_containers(my_instance, config)
    elif my_instance.df_host[it.host_field].nunique() < config['objective']['open_nodes']:
        print('Less open nodes than the objective !')
        satisfied = True
    else:
        print('Right number of open nodes.')
        satisfied = True

    return satisfied


# TODO define max by container
# TODO change alloc only for containers in node > max
def change_max_bound(my_instance: Instance, config: Dict, min_time: int):
    """Change max possible resource usage by containers."""
    # max_ok = False
    max_goal = get_abs_goal_load(my_instance, config)
    current_max_by_node = get_max_by_node(my_instance.df_indiv)
    total_max = get_total_max(current_max_by_node)
    print(current_max_by_node, total_max)
    print('total resources (max) : ', total_max)
    total_to_remove = resources_to_remove(max_goal, current_max_by_node)
    print('total resources to remove : ', total_to_remove)

    # If we want to remove same amount from all containers
    # decrease_cont = round_decimals_up(
    #     total_to_remove / my_instance.df_indiv[
    #         it.indiv_field].nunique())
    # print('resources to remove by container (average) : ', decrease_cont)

    # Remove resources at prorata of their initial max
    total_will_remove = 0.0
    print('Before change dataset : ')
    print(my_instance.df_indiv)
    for container in my_instance.df_indiv[it.indiv_field].unique():
        print('Doing container %s' % container)
        node = my_instance.df_indiv.loc[my_instance.df_indiv[
            it.indiv_field] == container][it.host_field].to_numpy()[0]
        percent_total = current_max_by_node[node][container] / total_max
        to_remove_c = total_to_remove * percent_total
        total_will_remove += to_remove_c
        current_max_by_node[node][container] = current_max_by_node[
            node][container] - to_remove_c
        change_max_dataset(my_instance.df_indiv,
                           container, it.indiv_field,
                           current_max_by_node[node][container], it.metrics[0],
                           it.tick_field, min_time)
    print('After change dataset :')
    print(my_instance.df_indiv)
    # input()
    print('Will be remove : ', total_will_remove)

    # Retrieve small amount on all, until needed is reached
    # while not max_ok:
    #     print(current_max_by_node)
    #     input()
    #     decrease = 0.1
    #     for container in my_instance.df_indiv[it.indiv_field]:
    #         node = my_instance.df_indiv.loc[my_instance.df_indiv[
    #             it.indiv_field] == container][it.host_field].to_numpy()[0]
    #         current_max_by_node[node][container] = current_max_by_node[
    #             node][container] - (current_max_by_node[
    #                 node][container] * decrease)
    #         if is_max_goal_ok(current_max_by_node, max_goal):
    #             max_ok = True
    #             break

    print('New max list : ', current_max_by_node)
    print('max goal satisfied ? ', is_max_goal_ok(current_max_by_node, max_goal))


def change_df_max(my_instance: Instance, c_id: str, max_c: float):
    """Change the DataFrame (resources values) after changing max."""
    for time in my_instance.df_indiv[it.tick_field].unique():
        if my_instance.df_indiv.loc[
            (my_instance.df_indiv[it.tick_field] == time) & (
                my_instance.df_indiv[it.indiv_field] == c_id), it.metrics[0]
        ].to_numpy()[0] > max_c:
            my_instance.df_indiv.loc[
                (my_instance.df_indiv[it.tick_field] == time) & (
                    my_instance.df_indiv[it.indiv_field] == c_id), it.metrics[0]
            ] = max_c


def get_max_by_node(df_indiv: pd.DataFrame) -> Dict:
    """Get the max possible usage of every container, by node."""
    max_by_node = {}
    for node, node_data in df_indiv.groupby(it.host_field):
        node_max = {}
        for container, cont_data in node_data.groupby(node_data[it.indiv_field]):
            node_max[container] = max(cont_data[it.metrics[0]])
        max_by_node[node] = node_max
    return max_by_node


def get_total_max(max_by_node: Dict) -> float:
    """Get the total amount of max resources."""
    total_max = 0.0
    for node in max_by_node.keys():
        total_max += sum(max_by_node[node].values())
    return total_max


def is_max_goal_ok(current_max_by_node: Dict, max_goal: float) -> bool:
    """Check is the max usage objective is satisfied."""
    for node in current_max_by_node.keys():
        if sum(current_max_by_node[node].values()) > max_goal:
            print('Not ok')
            return False
    return True


# TODO what if different goal on different nodes ? Dict of nodes goal ?
def get_abs_goal_load(my_instance: Instance, config: Dict) -> float:
    """Get the load goal in absolute value."""
    return config['objective']['target_load_CPU'] * (
        my_instance.df_host_meta[it.metrics[0]].to_numpy()[0]
    )


# TODO what if several nodes in goal ?
def resources_to_remove(max_goal: float, max_by_node: Dict) -> float:
    """Compute the amount of resources to remove to reach the load goal."""
    return (get_total_max(max_by_node) - max_goal)


def round_decimals_up(number: float, decimals: int = 2):
    """Return a value rounded up to a specific number of decimal places."""
    if not isinstance(decimals, int):
        raise TypeError('decimal places must be an integer')
    elif decimals < 0:
        raise ValueError('decimal places has to be 0 or more')
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def move_containers(my_instance: Instance, config: Dict):
    """Move containers in order to satisfy number open nodes target."""
    conso_nodes = np.zeros((
        config['objective']['open_nodes'], my_instance.window_duration))
    spread_containers(
        my_instance.df_indiv[it.indiv_field].unique(),
        my_instance, conso_nodes, my_instance.window_duration,
        config['objective']['open_nodes'])
