"""
=========
rac allocation
=========

Provide allocation heuristics and all allocation-related methods
(check capacity, change assignment ...).
Actually, there are 3 differents allocation techniques :
    - ``allocation_distant_pairwise`` : try to co-localize containers
    belonging to ``distant`` clusters ;
    - ``allocation_ffd`` : based on FFD bin-packing technique
    - spread : classical spread over nodes technique
"""

import math
from typing import List

import numpy as np
from tqdm import tqdm

from .instance import Instance


#####################################################################
# Functions definitions


def assign_container_node(node_id: str, container_id: str, instance: Instance):
    """Assign container_id to node_id, and remove it from old node."""
    old_id = instance.df_containers.loc[
        instance.df_containers['container_id'] == container_id
    ].machine_id.to_numpy()[0]

    instance.df_nodes.loc[
        instance.df_nodes['machine_id'] == node_id, ['cpu', 'mem']
    ] = instance.df_nodes.loc[
        instance.df_nodes['machine_id'] == node_id, ['cpu', 'mem']
    ].to_numpy() + instance.df_containers.loc[
        instance.df_containers['container_id'] == container_id, ['cpu', 'mem']
    ].to_numpy()

    instance.df_nodes.loc[
        instance.df_nodes['machine_id'] == old_id, ['cpu', 'mem']
    ] = instance.df_nodes.loc[
        instance.df_nodes['machine_id'] == old_id, ['cpu', 'mem']
    ].to_numpy() - instance.df_containers.loc[
        instance.df_containers['container_id'] == container_id, ['cpu', 'mem']
    ].to_numpy()

    instance.df_containers.loc[
        instance.df_containers['container_id'] == container_id, ['machine_id']
    ] = node_id


def allocation_distant_pairwise(
        instance: Instance, cluster_var_matrix: np.array,
        labels_: List, lb: float = 0.0) -> List:
    """
    First placement heuristic implemented : take two most distant clusters
    (from their mean profile), and assign by pair (one container from each
    cluster) on the first available node, and so on. Take the following node if
    current one has not enough resources.
    Return a list of containers forced to be grouped together.
    """
    print('Beginning of allocation ...')
    stop = False

    total_time = instance.sep_time

    min_nodes = nb_min_nodes(instance, total_time)
    conso_nodes = np.zeros((min_nodes, total_time))
    n = 0

    cluster_var_matrix_copy = np.copy(cluster_var_matrix)
    clusters_done_ = np.zeros(instance.nb_clusters, dtype=int)
    c_it = instance.nb_clusters

    pbar = tqdm(total=instance.nb_containers)
    containers_grouped = []

    while not stop:

        if (c_it == 0):
            stop = True
            break

        elif (c_it == 1):
            c = np.where(clusters_done_ == 0)[0][0]
            list_containers = [instance.dict_id_c[u] for u, value in enumerate(
                labels_) if value == c]
            for it in list_containers:
                cons_c = instance.df_containers.loc[
                    instance.df_containers['container_id'] == it
                ]['cpu'].to_numpy()[:total_time]
                cap_node = instance.df_nodes_meta.loc[
                    instance.df_nodes_meta['machine_id'] ==
                    instance.dict_id_n[n]
                ]['cpu'].to_numpy()[0]
                done = False
                while not done:
                    # TODO check n <= min_nodes or infeasibility
                    if np.all(np.less((conso_nodes[n] + cons_c), cap_node)):
                        conso_nodes[n] += cons_c
                        done = True
                        assign_container_node(
                            instance.dict_id_n[n], it, instance)
                        pbar.update(1)
                    else:
                        n = (n + 1) % min_nodes
                        cap_node = instance.df_nodes_meta.loc[
                            instance.df_nodes_meta['machine_id'] ==
                            instance.dict_id_n[n]
                        ]['cpu'].to_numpy()[0]
                n = (n + 1) % min_nodes
            stop = True

        else:
            valid_idx = np.where(cluster_var_matrix_copy.flatten() > lb)[0]
            min_idx = valid_idx[cluster_var_matrix_copy.flatten()[
                valid_idx].argmin()]
            i, j = np.unravel_index(
                min_idx, cluster_var_matrix_copy.shape)

            list_containers_i = [
                instance.dict_id_c[u] for u, value in
                enumerate(labels_) if value == i]
            list_containers_j = [
                instance.dict_id_c[u] for u, value in
                enumerate(labels_) if value == j]

            for it in range(min(len(list_containers_i),
                                len(list_containers_j))):
                cons_i = instance.df_containers.loc[
                    instance.df_containers['container_id'] ==
                    list_containers_i[it]
                ]['cpu'].to_numpy()[:total_time]
                cons_j = instance.df_containers.loc[
                    instance.df_containers['container_id'] ==
                    list_containers_j[it]
                ]['cpu'].to_numpy()[:total_time]

                cap_node = instance.df_nodes_meta.loc[
                    instance.df_nodes_meta['machine_id'] ==
                    instance.dict_id_n[n]
                ]['cpu'].to_numpy()[0]
                done = False
                while not done:
                    if np.all(np.less(
                            (conso_nodes[n] + cons_i + cons_j), cap_node)):
                        conso_nodes[n] += cons_i + cons_j
                        done = True
                        assign_container_node(
                            instance.dict_id_n[n],
                            list_containers_i[it],
                            instance)
                        assign_container_node(
                            instance.dict_id_n[n],
                            list_containers_j[it],
                            instance)
                        pbar.update(2)
                    else:
                        n = (n + 1) % min_nodes
                        cap_node = instance.df_nodes_meta.loc[
                            instance.df_nodes_meta['machine_id'] ==
                            instance.dict_id_n[n]
                        ]['cpu'].to_numpy()[0]
            # TODO factorization of container allocation
            if not (len(list_containers_i) == len(list_containers_j)):
                # we have to place remaining containers
                it += 1
                if it < len(list_containers_j):
                    for it in range(it, len(list_containers_j)):
                        cons_c = instance.df_containers.loc[
                            instance.df_containers['container_id'] ==
                            list_containers_j[it]
                        ]['cpu'].to_numpy()[:total_time]

                        if n is None:
                            n = 0
                        else:
                            n = (n + 1) % min_nodes
                        cap_node = instance.df_nodes_meta.loc[
                            instance.df_nodes_meta['machine_id'] ==
                            instance.dict_id_n[n]
                        ]['cpu'].to_numpy()[0]
                        done = False
                        while not done:
                            # TODO check n <= min_nodes or infeasibility
                            if np.all(np.less(
                                    (conso_nodes[n] + cons_c), cap_node)):
                                conso_nodes[n] += cons_c
                                done = True
                                assign_container_node(
                                    instance.dict_id_n[n],
                                    list_containers_j[it],
                                    instance)
                                pbar.update(1)
                            else:
                                n = (n + 1) % min_nodes
                                cap_node = instance.df_nodes_meta.loc[
                                    instance.df_nodes_meta['machine_id'] ==
                                    instance.dict_id_n[n]
                                ]['cpu'].to_numpy()[0]

                elif it < len(list_containers_i):
                    for it in range(it, len(list_containers_i)):
                        cons_c = instance.df_containers.loc[
                            instance.df_containers['container_id'] ==
                            list_containers_i[it]
                        ]['cpu'].to_numpy()[:total_time]

                        if n is None:
                            n = 0
                        else:
                            n = (n + 1) % min_nodes
                        cap_node = instance.df_nodes_meta.loc[
                            instance.df_nodes_meta['machine_id'] ==
                            instance.dict_id_n[n]
                        ]['cpu'].to_numpy()[0]
                        done = False
                        while not done:
                            # TODO check n <= min_nodes or infeasibility
                            if np.all(np.less(
                                    (conso_nodes[n] + cons_c), cap_node)):
                                conso_nodes[n] += cons_c
                                done = True
                                assign_container_node(
                                    instance.dict_id_n[n],
                                    list_containers_i[it],
                                    instance)
                                pbar.update(1)
                            else:
                                n = (n + 1) % min_nodes
                                cap_node = instance.df_nodes_meta.loc[
                                    instance.df_nodes_meta['machine_id'] ==
                                    instance.dict_id_n[n]
                                ]['cpu'].to_numpy()[0]
            containers_grouped.append(list_containers_i + list_containers_j)
            cluster_var_matrix_copy[i, :] = 0.0
            cluster_var_matrix_copy[:, i] = 0.0
            cluster_var_matrix_copy[j, :] = 0.0
            cluster_var_matrix_copy[:, j] = 0.0
            clusters_done_[i] = 1
            clusters_done_[j] = 1
            c_it = c_it - 2
    pbar.close()
    return containers_grouped


def allocation_ffd(instance: Instance,
                   cluster_vars: np.array, cluster_var_matrix: np.array,
                   labels_: List, bound_new_node: float = 50):
    """
    Second placement heuristic, based on "first-fit decreasing" bin-packing
    heuristic : order clusters by decreasing variance, place all containers
    belonging to the clusters in this order.
    For each container, try to place it on a node decreasing the node's
    variance. If not possible, place it on the node whose variance increases
    the least.
    """

    # TODO add open-nodes system (to run through open nodes only, and open a
    # new one when needed / with criterion)

    total_time = instance.sep_time

    # find minimum number of nodes needed
    min_nodes = nb_min_nodes(instance, total_time)

    idx_cluster_vars = np.argsort(cluster_vars)[::-1]
    conso_nodes = np.zeros((min_nodes, total_time))

    # try to place "opposite clusters" first
    conso_nodes, cluster_done = place_opposite_clusters(
        instance, cluster_vars,
        cluster_var_matrix, labels_,
        min_nodes, conso_nodes)

    nodes_vars = conso_nodes.var(axis=1)
    idx_nodes_vars = np.argsort(nodes_vars)[::-1]

    pbar = tqdm(total=instance.nb_containers)
    for cluster in np.where(cluster_done == 1)[0]:
        pbar.update(np.count_nonzero(labels_ == cluster))

    # We browse the clusters by variance decreasing
    for i in range(instance.nb_clusters):
        # check if cluster i has not be placed in init
        if not cluster_done[idx_cluster_vars[i]]:
            # print("We are in cluster ", idx_cluster_vars[i])
            list_containers = [instance.dict_id_c[j] for j, value in enumerate(
                labels_) if value == idx_cluster_vars[i]]

            # We browse containers in cluster i
            for container in list_containers:
                # print("We assign container ", container)
                consu_cont = instance.df_containers.loc[
                    instance.df_containers['container_id'] ==
                    container
                ]['cpu'].to_numpy()[:total_time]
                idx_node = 0
                min_var = math.inf
                idx_min_var = -1
                assign_container = 0

                # we run through already open nodes
                for idx_node in idx_nodes_vars:
                    # print("We try node ",
                    #       instance.dict_id_n[idx_node], idx_node)
                    cap_node = instance.df_nodes_meta.loc[
                        instance.df_nodes_meta['machine_id'] ==
                        instance.dict_id_n[idx_node]
                    ]['cpu'].to_numpy()[0]
                    new_var = (
                        conso_nodes[idx_node] + consu_cont).var()

                    # here we drop the node variance => ok
                    if (new_var <= nodes_vars[idx_node]) and np.all(
                        np.less(
                            (conso_nodes[idx_node] + consu_cont), cap_node)):

                        assign_container = idx_node + 1
                        break

                    # looking for the node variance grows the least
                    else:
                        if (new_var < min_var) and np.all(
                            np.less(
                                (conso_nodes[idx_node] + consu_cont),
                                cap_node)):
                            min_var = new_var
                            idx_min_var = idx_node

                if not assign_container:
                    if min_var < math.inf:
                        assign_container = idx_min_var + 1
                    # TODO here criterion for opening new node
                    # if criterion : open node
                    # we open a new node
                    else:
                        print('Critic open new node !')
                        print(conso_nodes)
                        print(consu_cont)
                        idx_node += 1
                    # TODO test nb_open nodes for feasability (raise error)

                # Finally we assign the container
                assign_container = assign_container - 1
                # print("Finally assign ", assign_container)
                conso_nodes[assign_container] += consu_cont
                nodes_vars[assign_container
                           ] = conso_nodes[assign_container].var()
                idx_nodes_vars = np.argsort(nodes_vars)[::-1]
                assign_container_node(
                    instance.dict_id_n[assign_container], container, instance)
                pbar.update(1)
    pbar.close()


# TODO adapt to dfs
# Spread technique for allocation
# def allocation_spread(instance, nodes_available):

#     services_done_ = np.zeros(instance.nb_services)
#     assign_s = 0
#     i_s = 0
#     for service in instance.services_:
#         assign = False

#         while not assign:
#             if not enough_resource(
#                     nodes_available[assign_s, :, :],
#                     service, instance.nodes_init[assign_s].ram,
#                     instance.nodes_init[assign_s].cpu):
#                 assign_s = (assign_s + 1) % instance.nb_nodes
#             else:
#                 assign = True
#                 assign_service_node(nodes_available[assign_s, :, :], service)
#         services_done_[i_s] = assign_s + 1
#         assign_s = (assign_s + 1) % instance.nb_nodes
#         i_s = i_s + 1

#     return (services_done_)


# TODO consider 85% usage now ?
# Try to find minimum number of nodes needed
def nb_min_nodes(instance: Instance, total_time: int) -> (float, float):
    max_cpu = 0.0
    max_mem = 0.0
    for t in range(total_time):
        max_t_cpu = instance.df_containers[
            instance.df_containers['timestamp'] == t]['cpu'].sum()
        max_t_mem = instance.df_containers[
            instance.df_containers['timestamp'] == t]['mem'].sum()

        if max_t_cpu > max_cpu:
            max_cpu = max_t_cpu
        if max_t_mem > max_mem:
            max_mem = max_t_mem

    # TODO consider nodes with different capacities
    cap_cpu = instance.df_nodes_meta['cpu'].to_numpy()[0]
    cap_mem = instance.df_nodes_meta['mem'].to_numpy()[0]
    min_nodes_cpu = math.ceil(max_cpu / cap_cpu)
    min_nodes_mem = math.ceil(max_mem / cap_mem)
    return max(min_nodes_cpu, min_nodes_mem)


# TODO integrate upper bound for considering all clusters sum variance < ub
def place_opposite_clusters(instance: Instance, cluster_vars: np.array,
                            cluster_var_matrix: np.array, labels_: List,
                            min_nodes: int, conso_nodes: np.array
                            ) -> (np.array, np.array):
    total_time = instance.sep_time
    lb = 0.0
    valid_idx = np.where(cluster_var_matrix.flatten() > lb)[0]
    min_idx = valid_idx[cluster_var_matrix.flatten()[valid_idx].argmin()]
    i, j = np.unravel_index(
        min_idx, cluster_var_matrix.shape)
    cluster_done = np.zeros(instance.nb_clusters)

    list_containers_i = [instance.dict_id_c[u] for u, value in enumerate(
        labels_) if value == i]
    list_containers_j = [instance.dict_id_c[u] for u, value in enumerate(
        labels_) if value == j]

    for it in range(min(len(list_containers_i), len(list_containers_j))):
        cons_i = instance.df_containers.loc[
            instance.df_containers['container_id'] ==
            list_containers_i[it]
        ]['cpu'].to_numpy()[:total_time]

        cons_j = instance.df_containers.loc[
            instance.df_containers['container_id'] ==
            list_containers_j[it]
        ]['cpu'].to_numpy()[:total_time]

        n = 0
        cap_node = instance.df_nodes_meta.loc[
            instance.df_nodes_meta['machine_id'] ==
            instance.dict_id_n[n]
        ]['cpu'].to_numpy()[0]
        done = False
        while not done:
            # TODO check n <= min_nodes or infeasibility
            if np.all(np.less((conso_nodes[n] + cons_i + cons_j), cap_node)):
                conso_nodes[n] += cons_i + cons_j
                done = True
                assign_container_node(
                    instance.dict_id_n[n], list_containers_i[it], instance)
                assign_container_node(
                    instance.dict_id_n[n], list_containers_j[it], instance)
            else:
                n = (n + 1) % min_nodes
                cap_node = instance.df_nodes_meta.loc[
                    instance.df_nodes_meta['machine_id'] ==
                    instance.dict_id_n[n]
                ]['cpu'].to_numpy()[0]

    # TODO factorization of container allocation
    if not (len(list_containers_i) == len(list_containers_j)):
        # we have to place remaining containers

        if it < len(list_containers_j):
            print('Remaining %d containers' % (len(list_containers_j) - it))
            for it in range(it, len(list_containers_j)):
                cons_c = instance.df_containers.loc[
                    instance.df_containers['container_id'] ==
                    list_containers_j[it]
                ]['cpu'].to_numpy()[:total_time]

                if n is None:
                    n = 0
                else:
                    n = (n + 1) % min_nodes
                cap_node = instance.df_nodes_meta.loc[
                    instance.df_nodes_meta['machine_id'] ==
                    instance.dict_id_n[n]
                ]['cpu'].to_numpy()[0]
                done = False
                while not done:
                    # TODO check n <= min_nodes or infeasibility
                    if np.all(np.less((conso_nodes[n] + cons_c), cap_node)):
                        conso_nodes[n] += cons_c
                        done = True
                        assign_container_node(
                            instance.dict_id_n[n],
                            list_containers_j[it],
                            instance)
                    else:
                        n = (n + 1) % min_nodes
                        cap_node = instance.df_nodes_meta.loc[
                            instance.df_nodes_meta['machine_id'] ==
                            instance.dict_id_n[n]
                        ]['cpu'].to_numpy()[0]

        elif it < len(list_containers_i):
            for it in range(it, len(list_containers_i)):
                cons_c = instance.df_containers.loc[
                    instance.df_containers['container_id'] ==
                    list_containers_i[it]
                ]['cpu'].to_numpy()[:total_time]

                if n is None:
                    n = 0
                else:
                    n = (n + 1) % min_nodes
                cap_node = instance.df_nodes_meta.loc[
                    instance.df_nodes_meta['machine_id'] ==
                    instance.dict_id_n[n]
                ]['cpu'].to_numpy()[0]
                done = False
                while not done:
                    # TODO check n <= min_nodes or infeasibility
                    if np.all(np.less((conso_nodes[n] + cons_c), cap_node)):
                        conso_nodes[n] += cons_c
                        done = True
                        assign_container_node(
                            instance.dict_id_n[n],
                            list_containers_i[it],
                            instance)
                    else:
                        n = (n + 1) % min_nodes
                        cap_node = instance.df_nodes_meta.loc[
                            instance.df_nodes_meta['machine_id'] ==
                            instance.dict_id_n[n]
                        ]['cpu'].to_numpy()[0]

    cluster_done[i] = 1
    cluster_done[j] = 1
    return conso_nodes, cluster_done
