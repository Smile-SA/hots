"""Provide resource allocation related functions to handle this problem."""

import math

import numpy as np

from . import init as it
from .placement import spread_containers
from .tools import change_max_dataset


def check_constraints(
    my_instance, working_df_indiv, config
):
    """Check if allocation constraints are satisfied or not.

    :param my_instance: The Instance object of the current run
    :type my_instance: Instance
    :param working_df_indiv: Dataframe with current window individual data
    :type working_df_indiv: pd.DataFrame
    :param config: Current HOTS run parameters
    :type config: Dict
    :return: True if all constraints are satisfied, False otherwise
    :rtype: bool
    """
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
def change_max_bound(my_instance, config, min_time):
    """Change max possible resource usage by containers.

    :param my_instance: The Instance object of the current run
    :type my_instance: Instance
    :param config: Current HOTS run parameters
    :type config: Dict
    :param min_time: T0 of current time window
    :type min_time: int
    """
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


def change_df_max(my_instance, c_id, max_c):
    """Change the DataFrame (resources values) after changing max.

    :param my_instance: The Instance object of the current run
    :type my_instance: Instance
    :param c_id: ID of container to change values
    :type c_id: str
    :param max_c: New bound value for c_id data
    :type max_c: float
    """
    for time in my_instance.df_indiv[it.tick_field].unique():
        if my_instance.df_indiv.loc[
            (my_instance.df_indiv[it.tick_field] == time) & (
                my_instance.df_indiv[it.indiv_field] == c_id), it.metrics[0]
        ].to_numpy()[0] > max_c:
            my_instance.df_indiv.loc[
                (my_instance.df_indiv[it.tick_field] == time) & (
                    my_instance.df_indiv[it.indiv_field] == c_id), it.metrics[0]
            ] = max_c


def get_max_by_node(df_indiv):
    """Get the max possible usage of every container, by node.

    :param df_indiv: Dataframe storing individual data
    :type df_indiv: pd.DataFrame
    :return: Max container value for each node
    :rtype: Dict
    """
    max_by_node = {}
    for node, node_data in df_indiv.groupby(it.host_field):
        node_max = {}
        for container, cont_data in node_data.groupby(node_data[it.indiv_field]):
            node_max[container] = max(cont_data[it.metrics[0]])
        max_by_node[node] = node_max
    return max_by_node


def get_total_max(max_by_node):
    """Get the total amount of max resources.

    :param max_by_node: Dict object of max container value in each node
    :type max_by_node: Dict
    :return: Sum of every container max value from each node
    :rtype: float
    """
    total_max = 0.0
    for node in max_by_node.keys():
        total_max += sum(max_by_node[node].values())
    return total_max


def is_max_goal_ok(current_max_by_node, max_goal):
    """Check is the max usage objective is satisfied.

    :param current_max_by_node: Max container value for each node
    :type current_max_by_node: Dict
    :param max_goal: Max value objective (not to exceed)
    :type max_goal: float
    :return: True if objective is satisfied, False otherwise
    :rtype: bool
    """
    for node in current_max_by_node.keys():
        if sum(current_max_by_node[node].values()) > max_goal:
            print('Not ok')
            return False
    return True


# TODO what if different goal on different nodes ? Dict of nodes goal ?
def get_abs_goal_load(my_instance, config):
    """Get the load goal in absolute value.

    :param my_instance: The Instance object of the current run
    :type my_instance: Instance
    :param config: Current HOTS run parameters
    :type config: Dict
    :return: Load value to achieve from parameters
    :rtype: float
    """
    return config['objective']['target_load_CPU'] * (
        my_instance.df_host_meta[it.metrics[0]].to_numpy()[0]
    )


# TODO what if several nodes in goal ?
def resources_to_remove(max_goal, max_by_node):
    """Compute the amount of resources to remove to reach the load goal.

    :param max_goal: Load value to achieve from parameters
    :type max_goal: float
    :param max_by_node: Max container value for each node
    :type max_by_node: Dict
    :return: Value to retrieve for reaching load goal
    :rtype: float
    """
    return (get_total_max(max_by_node) - max_goal)


def round_decimals_up(number, decimals=2):
    """Return a value rounded up to a specific number of decimal places.

    :param number: Value to round up
    :type number: float
    :param decimals: Wanted numbers after comma, defaults to 2
    :type decimals: int, optional
    :raises TypeError: Wrong type for decimals
    :raises ValueError: Non-positive value for decimals
    :return: Rounded up value
    :rtype: float
    """
    if not isinstance(decimals, int):
        raise TypeError('decimal places must be an integer')
    elif decimals < 0:
        raise ValueError('decimal places has to be 0 or more')
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def move_containers(my_instance, config):
    """Move containers in order to satisfy number open nodes target.

    :param my_instance: The Instance object of the current run
    :type my_instance: Instance
    :param config: Current HOTS run parameters
    :type config: Dict
    """
    conso_nodes = np.zeros((
        config['objective']['open_nodes'], my_instance.window_duration))
    spread_containers(
        my_instance.df_indiv[it.indiv_field].unique(),
        my_instance, conso_nodes, my_instance.window_duration,
        config['objective']['open_nodes'])
