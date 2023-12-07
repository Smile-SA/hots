"""Provide placement heuristics and all placement-related methods."""

import math
import random
from itertools import combinations

import numpy as np

from . import init as it

#####################################################################
# Functions definitions


# TODO factorize many functions (assign indiv, change node during assign indiv..)
def assign_container_node(node_id, container_id, instance, remove=True):
    """Assign container_id to node_id, and remove it from old node.

    :param node_id: _description_
    :type node_id: str
    :param container_id: _description_
    :type container_id: str
    :param instance: _description_
    :type instance: Instance
    :param remove: _description_, defaults to True
    :type remove: bool, optional
    """
    old_id = instance.df_indiv.loc[
        instance.df_indiv[it.indiv_field] == container_id
    ][it.host_field].to_numpy()[0]

    instance.df_host.loc[
        instance.df_host[it.host_field] == node_id, it.metrics
    ] = instance.df_host.loc[
        instance.df_host[it.host_field] == node_id, it.metrics
    ].to_numpy() + instance.df_indiv.loc[
        instance.df_indiv[it.indiv_field] == container_id, it.metrics
    ].to_numpy()

    if remove:
        remove_container_node(old_id, container_id, instance)

    instance.df_indiv.loc[
        instance.df_indiv[it.indiv_field] == container_id, [it.host_field]
    ] = node_id


def remove_container_node(node_id, container_id, instance):
    """Remove container from node.

    :param node_id: _description_
    :type node_id: str
    :param container_id: _description_
    :type container_id: str
    :param instance: _description_
    :type instance: Instance
    """
    instance.df_host.loc[
        instance.df_host[it.host_field] == node_id, it.metrics
    ] = instance.df_host.loc[
        instance.df_host[it.host_field] == node_id, it.metrics
    ].to_numpy() - instance.df_indiv.loc[
        instance.df_indiv[it.indiv_field] == container_id, it.metrics
    ].to_numpy()


def free_full_nodes(instance, full_nodes, tick):
    """Change the solution in order to satisfy node capacities.

    :param instance: _description_
    :type instance: Instance
    :param full_nodes: _description_
    :type full_nodes: List
    :param tick: _description_
    :type tick: int
    """
    for host in full_nodes:
        stop = False
        while not stop:
            df_indiv_host = instance.df_indiv.loc[
                (instance.df_indiv[it.host_field] == host) & (
                    instance.df_indiv[it.tick_field] == tick
                )
            ]
            if df_indiv_host[it.metrics[0]].sum() > instance.df_host_meta.loc[
                instance.df_host_meta[it.host_field] == host
            ][it.metrics[0]].to_numpy()[0]:
                moving_indiv = random.choice(
                    df_indiv_host[it.indiv_field].unique())
                assign_indiv_available_host(
                    instance, moving_indiv, tick, tick,
                    instance.df_indiv[it.host_field].nunique())
            else:
                stop = True


def assign_indiv_available_host(instance, indiv_id, tmin, tmax, nb_open_nodes):
    """Assign the individual to first available host.

    :param instance: _description_
    :type instance: Instance
    :param indiv_id: _description_
    :type indiv_id: str
    :param tmin: _description_
    :type tmin: int
    :param tmax: _description_
    :type tmax: int
    :param nb_open_nodes: _description_
    :type nb_open_nodes: int
    :raises RuntimeError: _description_
    """
    print('Moving individual %s' % indiv_id)
    cons_c = instance.df_indiv.loc[
        (instance.df_indiv[it.indiv_field] == indiv_id) & (
            instance.df_indiv[it.tick_field] >= tmin) & (
            instance.df_indiv[it.tick_field] <= tmax
        )
    ][it.metrics[0]].to_numpy()

    n = 0
    checked_nodes = 1
    done = False
    while not done:
        # TODO check n <= min_nodes or infeasibility
        cap_node = instance.df_host_meta.loc[
            instance.
            df_host_meta[it.host_field] == instance.dict_id_n[n]
        ][it.metrics[0]].to_numpy()[0]
        conso_node = instance.df_indiv.loc[
            (instance.df_indiv[it.host_field] == instance.dict_id_n[n]) & (
                instance.df_indiv[it.tick_field] >= tmin) & (
                instance.df_indiv[it.tick_field] <= tmax
            )
        ][it.metrics[0]].sum()
        if np.all(np.less((conso_node + cons_c), cap_node)):
            conso_node += cons_c
            done = True
            assign_container_node(
                instance.dict_id_n[n], indiv_id, instance)
        else:
            checked_nodes += 1
            n = (n + 1) % nb_open_nodes
            if checked_nodes > nb_open_nodes:
                print('Impossible to move %s on another existing node.' % indiv_id)
                if (instance.dict_id_n[checked_nodes] in
                    instance.df_host_meta[it.host_field].unique()) & (
                        np.all(np.less(cons_c, cap_node))):
                    print('We can open node %s' % (instance.dict_id_n[checked_nodes]))
                    assign_container_node(
                        instance.dict_id_n[checked_nodes], indiv_id, instance)
                    done = True
                else:
                    raise RuntimeError('No node to welcome %s' % indiv_id)
                # find_substitution(instance, indiv_id, tmin, tmax)


def assign_indiv_initial_placement(
    instance, indiv_id, tmin, tmax, conso_nodes, min_nodes, n=0
):
    """Assign indiv in node during first heuristic.

    :param instance: _description_
    :type instance: Instance
    :param indiv_id: _description_
    :type indiv_id: str
    :param tmin: _description_
    :type tmin: int
    :param tmax: _description_
    :type tmax: int
    :param conso_nodes: _description_
    :type conso_nodes: List
    :param min_nodes: _description_
    :type min_nodes: int
    :param n: _description_, defaults to 0
    :type n: int, optional
    :return: _description_
    :rtype: bool
    """
    cons_c = instance.df_indiv.loc[
        (instance.df_indiv[it.indiv_field] == indiv_id) & (
            instance.df_indiv[it.tick_field] >= tmin) & (
            instance.df_indiv[it.tick_field] <= tmax
        )
    ][it.metrics[0]].to_numpy()
    checked_nodes = 1
    done = False
    while not done:
        cap_node = instance.df_host_meta.loc[
            instance.
            df_host_meta[it.host_field] == instance.dict_id_n[n]
        ][it.metrics[0]].to_numpy()[0]
        if np.all(np.less(
                (conso_nodes[n] + cons_c), cap_node)):
            conso_nodes[n] += cons_c
            assign_container_node(
                instance.dict_id_n[n],
                indiv_id,
                instance)
            done = True
        else:
            checked_nodes += 1
            if checked_nodes >= min_nodes:
                print('Impossible to move %s on another existing node.' % indiv_id)
                if (instance.dict_id_n[checked_nodes] in
                    instance.df_host_meta[it.host_field].unique()) & (
                        np.all(np.less(cons_c, cap_node))):
                    print('We can open node %s' % (instance.dict_id_n[checked_nodes]))
                    min_nodes += 1
                    assign_container_node(
                        instance.dict_id_n[checked_nodes],
                        indiv_id,
                        instance)
                    done = True
            else:
                n = (n + 1) % min_nodes
            cap_node = instance.df_host_meta.loc[
                instance.
                df_host_meta[it.host_field] == instance.dict_id_n[n]
            ][it.metrics[0]].to_numpy()[0]
    return done


def find_substitution(instance, indiv_id, tmin, tmax):
    """Find a node in which we can place indiv_id by moving another indiv.

    :param instance: _description_
    :type instance: Instance
    :param indiv_id: _description_
    :type indiv_id: str
    :param tmin: _description_
    :type tmin: int
    :param tmax: _description_
    :type tmax: int
    :raises RuntimeError: _description_
    """
    # TODO not fully functionnal
    cons_c = instance.df_indiv.loc[
        (instance.df_indiv[it.indiv_field] == indiv_id) & (
            instance.df_indiv[it.tick_field] >= tmin) & (
            instance.df_indiv[it.tick_field] <= tmax
        )
    ][it.metrics[0]].to_numpy()

    n = 0
    checked_nodes = 1
    done = False
    while not done:
        conso_node = instance.df_indiv.loc[
            (instance.df_indiv[it.host_field] == instance.dict_id_n[n]) & (
                instance.df_indiv[it.tick_field] >= tmin) & (
                instance.df_indiv[it.tick_field] <= tmax
            )
        ][it.metrics[0]].sum()
        cap_node = instance.df_host_meta.loc[
            instance.
            df_host_meta[it.host_field] == instance.dict_id_n[n]
        ][it.metrics[0]].to_numpy()[0]
        moving_indiv = random.choice(
            instance.df_indiv.loc[
                (instance.df_indiv[it.host_field] == instance.dict_id_n[n])
            ][it.indiv_field].unique())
        conso_indiv = instance.df_indiv.loc[
            (instance.df_indiv[it.indiv_field] == moving_indiv) & (
                instance.df_indiv[it.tick_field] >= tmin) & (
                instance.df_indiv[it.tick_field] <= tmax)
        ][it.metrics[0]].to_numpy()
        if np.all(np.less((conso_node - conso_indiv + cons_c), cap_node)):
            done = True
            assign_container_node(
                instance.dict_id_n[n], indiv_id, instance)
            assign_indiv_available_host(instance, moving_indiv, tmin, tmax)
        else:
            checked_nodes += 1
            n = (n + 1) % instance.nb_nodes
            if checked_nodes > instance.nb_nodes:
                raise RuntimeError('No node to welcome %s' % indiv_id)


def spread_containers_new(list_containers, instance, conso_nodes, total_time, min_nodes):
    """Propose an alternative to spread technique."""
    df_indiv = instance.df_indiv
    df_host_meta = instance.df_host_meta
    dict_id_n = instance.dict_id_n
    host_field = it.host_field
    indiv_field = it.indiv_field
    metrics_0 = it.metrics[0]
    # tick_field_min = instance.df_indiv[it.tick_field].min()

    n = 0
    checked_nodes = 1

    for c in list_containers:
        indiv_c = c
        cons_c = df_indiv[df_indiv[indiv_field] == indiv_c][metrics_0].to_numpy()[:total_time]
        cap_node = df_host_meta[df_host_meta[host_field] == dict_id_n[n]][metrics_0].to_numpy()[0]
        done = False

        while not done:
            if np.less(conso_nodes[n] + cons_c, cap_node).all():
                conso_nodes[n] += cons_c
                assign_container_node(dict_id_n[n], indiv_c, instance)
                done = True
            else:
                checked_nodes += 1
                if checked_nodes > min_nodes:
                    min_nodes += 1
                    checked_nodes = 1
                n = (n + 1) % min_nodes
                cap_node = df_host_meta[
                    df_host_meta[host_field] == dict_id_n[n]][metrics_0].to_numpy()[0]

        n = (n + 1) % min_nodes

    return n


def spread_containers(
    list_containers, instance, conso_nodes, total_time, min_nodes
):
    """Spread containers from list_containers into nodes.

    :param list_containers: _description_
    :type list_containers: List
    :param instance: _description_
    :type instance: Instance
    :param conso_nodes: _description_
    :type conso_nodes: np.array
    :param total_time: _description_
    :type total_time: int
    :param min_nodes: _description_
    :type min_nodes: int
    """
    n = 0
    for c in list_containers:
        cons_c = instance.df_indiv.loc[
            instance.df_indiv[it.indiv_field] == c
        ][it.metrics[0]].to_numpy()[:total_time]
        cap_node = instance.df_host_meta.loc[
            instance.
            df_host_meta[it.host_field] == instance.dict_id_n[n]
        ][it.metrics[0]].to_numpy()[0]
        checked_nodes = 1
        done = False
        while not done:
            # TODO check n <= min_nodes or infeasibility
            if np.all(np.less((conso_nodes[n] + cons_c), cap_node)):
                conso_nodes[n] += cons_c
                done = True
                assign_container_node(
                    instance.dict_id_n[n], c, instance)
            else:
                checked_nodes += 1
                if checked_nodes > min_nodes:
                    min_nodes += 1
                    checked_nodes = 1
                n = (n + 1) % min_nodes
                cap_node = instance.df_host_meta.loc[
                    instance.
                    df_host_meta[it.host_field] == instance.dict_id_n[n]
                ][it.metrics[0]].to_numpy()[0]
        n = (n + 1) % min_nodes


def colocalize_clusters(
    list_containers_i, list_containers_j, containers_grouped, instance, total_time, min_nodes,
    conso_nodes, n=0
):
    """Allocate containers of 2 clusters grouping by pairs.

    :param list_containers_i: _description_
    :type list_containers_i: List
    :param list_containers_j: _description_
    :type list_containers_j: List
    :param containers_grouped: _description_
    :type containers_grouped: List
    :param instance: _description_
    :type instance: Instance
    :param total_time: _description_
    :type total_time: int
    :param min_nodes: _description_
    :type min_nodes: int
    :param conso_nodes: _description_
    :type conso_nodes: List
    :param n: _description_, defaults to 0
    :type n: int, optional
    :raises RuntimeError: _description_
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: int
    """
    for c in range(min(len(list_containers_i),
                       len(list_containers_j))):
        # allocate 2 containers !! TODO
        cons_i = instance.df_indiv.loc[
            instance.
            df_indiv[it.indiv_field] == list_containers_i[c]
        ][it.metrics[0]].to_numpy()[:total_time]
        cons_j = instance.df_indiv.loc[
            instance.
            df_indiv[it.indiv_field] == list_containers_j[c]
        ][it.metrics[0]].to_numpy()[:total_time]

        cap_node = instance.df_host_meta.loc[
            instance.
            df_host_meta[it.host_field] == instance.dict_id_n[n]
        ][it.metrics[0]].to_numpy()[0]
        i_n = 0
        if not np.all(np.less(
                cons_i + cons_j, cap_node)):
            # If indivs i & j can't fit together : split them
            if not assign_indiv_initial_placement(
                    instance, list_containers_i[c],
                    instance.df_indiv[it.tick_field].min(), total_time - 1,
                    conso_nodes, min_nodes, n):
                raise RuntimeError('No node to welcome %s' % list_containers_i[c])
            n = (n + 1) % min_nodes
            if not assign_indiv_initial_placement(
                    instance, list_containers_j[c],
                    instance.df_indiv[it.tick_field].min(), total_time - 1,
                    conso_nodes, min_nodes, n):
                raise RuntimeError('No node to welcome %s' % list_containers_j[c])

        else:
            # Indivs i & j could fit together
            done = False
            while not done:
                if np.all(np.less(
                        (conso_nodes[n] + cons_i + cons_j), cap_node)):
                    conso_nodes[n] += cons_i + cons_j
                    assign_container_node(
                        instance.dict_id_n[n],
                        list_containers_i[c],
                        instance)
                    assign_container_node(
                        instance.dict_id_n[n],
                        list_containers_j[c],
                        instance)
                    containers_grouped.append([
                        list_containers_i[c], list_containers_j[c]])
                    done = True
                else:
                    i_n += 1
                    if i_n >= min_nodes:
                        min_nodes += 1
                        n = i_n
                    else:
                        n = (n + 1) % min_nodes
                    cap_node = instance.df_host_meta.loc[
                        instance.
                        df_host_meta[it.host_field] == instance.dict_id_n[n]
                    ][it.metrics[0]].to_numpy()[0]
            n = (n + 1) % min_nodes
    return c


def colocalize_clusters_new(
    list_containers_i, list_containers_j, containers_grouped, instance, total_time, min_nodes,
    conso_nodes, n=0
):
    """Allocate containers of 2 clusters grouping by pairs.

    :param list_containers_i: _description_
    :type list_containers_i: List
    :param list_containers_j: _description_
    :type list_containers_j: List
    :param containers_grouped: _description_
    :type containers_grouped: List
    :param instance: _description_
    :type instance: Instance
    :param total_time: _description_
    :type total_time: int
    :param min_nodes: _description_
    :type min_nodes: int
    :param conso_nodes: _description_
    :type conso_nodes: List
    :param n: _description_, defaults to 0
    :type n: int, optional
    :raises RuntimeError: _description_
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: int
    """
    df_indiv = instance.df_indiv
    df_host_meta = instance.df_host_meta
    dict_id_n = instance.dict_id_n

    for c in range(min(len(list_containers_i), len(list_containers_j))):
        indiv_i = list_containers_i[c]
        indiv_j = list_containers_j[c]

        cons_i = df_indiv.loc[
            df_indiv[it.indiv_field] == indiv_i
        ][it.metrics[0]].to_numpy()[:total_time]
        cons_j = df_indiv.loc[
            df_indiv[it.indiv_field] == indiv_j
        ][it.metrics[0]].to_numpy()[:total_time]

        host_field_n = dict_id_n[n]
        cap_node = df_host_meta.loc[
            df_host_meta[it.host_field] == host_field_n
        ][it.metrics[0]].to_numpy()[0]

        if not np.all(np.less(cons_i + cons_j, cap_node)):
            # If indivs i & j can't fit together: split them
            # start = time.time()
            if not assign_indiv_initial_placement(
                instance, indiv_i, df_indiv[it.tick_field].min(),
                total_time - 1, conso_nodes, min_nodes, n
            ):
                raise RuntimeError('No node to welcome %s' % indiv_i)
            n = (n + 1) % min_nodes
            if not assign_indiv_initial_placement(
                instance, indiv_j, df_indiv[it.tick_field].min(),
                total_time - 1, conso_nodes, min_nodes, n
            ):
                raise RuntimeError('No node to welcome %s' % indiv_j)
        else:
            # Indivs i & j could fit together
            done = False
            while not done:
                if np.all(np.less(conso_nodes[n] + cons_i + cons_j, cap_node)):
                    conso_nodes[n] += cons_i + cons_j
                    assign_container_node(dict_id_n[n], indiv_i, instance)
                    assign_container_node(dict_id_n[n], indiv_j, instance)
                    containers_grouped.append([indiv_i, indiv_j])
                    done = True
                else:
                    n = (n + 1) % min_nodes
                    if n >= min_nodes:
                        min_nodes += 1
                    cap_node = df_host_meta.loc[
                        df_host_meta[it.host_field] == dict_id_n[n]
                    ][it.metrics[0]].to_numpy()[0]
            n = (n + 1) % min_nodes
    return c


def allocation_distant_pairwise(
    instance, cluster_var_matrix, labels_, nb_nodes=None, lb=0.0
):
    """First placement heuristic implemented.

    Idea : take two most distant clusters
    (from their mean profile), and assign by pair (one container from each
    cluster) on the first available node, and so on. Take the following node if
    current one has not enough resources.
    Return a list of containers forced to be grouped together.

    :param instance: _description_
    :type instance: Instance
    :param cluster_var_matrix: _description_
    :type cluster_var_matrix: np.array
    :param labels_: _description_
    :type labels_: List
    :param nb_nodes: _description_, defaults to None
    :type nb_nodes: int, optional
    :param lb: _description_, defaults to 0.0
    :type lb: float, optional
    """
    print('Beginning of allocation ...')
    stop = False

    total_time = instance.sep_time
    min_nodes = nb_nodes or nb_min_nodes(instance, total_time)
    conso_nodes = np.zeros((instance.nb_nodes, total_time))
    n = 0

    cluster_var_matrix_copy = np.copy(cluster_var_matrix)
    clusters_done_ = np.zeros(instance.nb_clusters, dtype=int)
    c_it = instance.nb_clusters

    containers_grouped = []

    while not stop:
        # no cluster remaining -> stop allocation
        if (c_it == 0):
            stop = True
            break

        # 1 cluster remaining -> spread it TODO
        elif (c_it == 1):
            c = np.where(clusters_done_ == 0)[0][0]
            list_containers = [instance.dict_id_c[u] for u, value in enumerate(
                labels_) if value == c]
            spread_containers(list_containers, instance, conso_nodes,
                              total_time, min_nodes)
            stop = True

        # > 1 cluster remaining -> co-localize 2 more distant
        else:
            valid_idx = np.where(cluster_var_matrix_copy.flatten() >= lb)[0]
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
            it_cont = colocalize_clusters(list_containers_i, list_containers_j,
                                          containers_grouped, instance, total_time,
                                          min_nodes, conso_nodes, n)
            # TODO factorization of container allocation
            if not (len(list_containers_i) == len(list_containers_j)):
                # we have to place remaining containers
                it_cont += 1
                if it_cont < len(list_containers_j):
                    list_containers = list_containers_j[it_cont:]
                    spread_containers(list_containers, instance, conso_nodes,
                                      total_time, min_nodes)

                elif it_cont < len(list_containers_i):
                    list_containers = list_containers_i[it_cont:]
                    spread_containers(list_containers, instance, conso_nodes,
                                      total_time, min_nodes)

            cluster_var_matrix_copy[i, :] = -1.0
            cluster_var_matrix_copy[:, i] = -1.0
            cluster_var_matrix_copy[j, :] = -1.0
            cluster_var_matrix_copy[:, j] = -1.0
            clusters_done_[i] = 1
            clusters_done_[j] = 1
            c_it = c_it - 2


def allocation_ffd(
    instance, cluster_vars, cluster_var_matrix, labels_, bound_new_node=50
):
    """Second placement heuristic.

    Idea (based on "first-fit decreasing" bin-packing
    heuristic) : order clusters by decreasing variance, place all containers
    belonging to the clusters in this order.
    For each container, try to place it on a node decreasing the node's
    variance. If not possible, place it on the node whose variance increases
    the least.

    :param instance: _description_
    :type instance: Instance
    :param cluster_vars: _description_
    :type cluster_vars: np.array
    :param cluster_var_matrix: _description_
    :type cluster_var_matrix: np.array
    :param labels_: _description_
    :type labels_: List
    :param bound_new_node: _description_, defaults to 50
    :type bound_new_node: float, optional
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

    # We browse the clusters by variance decreasing
    for i in range(instance.nb_clusters):
        # check if cluster i has not be placed in init
        if not cluster_done[idx_cluster_vars[i]]:
            list_containers = [instance.dict_id_c[j] for j, value in enumerate(
                labels_) if value == idx_cluster_vars[i]]

            # We browse containers in cluster i
            for container in list_containers:
                consu_cont = instance.df_indiv.loc[
                    instance.
                    df_indiv[it.indiv_field] == container
                ][it.metrics[0]].to_numpy()[:total_time]
                idx_node = 0
                min_var = math.inf
                idx_min_var = -1
                assign_container = 0

                # we run through already open nodes
                for idx_node in idx_nodes_vars:
                    cap_node = instance.df_host_meta.loc[
                        instance.
                        df_host_meta[it.host_field] == instance.dict_id_n[idx_node]
                    ][it.metrics[0]].to_numpy()[0]
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
                        idx_node += 1
                    # TODO test nb_open nodes for feasability (raise error)

                # Finally we assign the container
                assign_container = assign_container - 1
                conso_nodes[assign_container] += consu_cont
                nodes_vars[assign_container
                           ] = conso_nodes[assign_container].var()
                idx_nodes_vars = np.argsort(nodes_vars)[::-1]
                assign_container_node(
                    instance.dict_id_n[assign_container], container, instance)


def allocation_spread(instance, min_nodes=None):
    """Spread technique for placement.

    :param instance: _description_
    :type instance: Instance
    :param min_nodes: _description_, defaults to None
    :type min_nodes: int, optional
    """
    total_time = instance.sep_time
    min_nodes = min_nodes or nb_min_nodes(instance, total_time)
    conso_nodes = np.zeros((instance.nb_nodes, total_time))
    spread_containers(
        instance.df_indiv[it.indiv_field].unique(),
        instance, conso_nodes, total_time, min_nodes
    )


# TODO consider 85% usage now ? maybe add parameter
# TODO make it generic with metrics
def nb_min_nodes(instance, total_time):
    """Compute the minimum number of nodes needed to support the load.

    :param instance: _description_
    :type instance: Instance
    :param total_time: _description_
    :type total_time: int
    :return: _description_
    :rtype: float
    """
    # max_cpu = 0.0
    # max_mem = 0.0
    # for t in range(total_time):
    #     max_t_cpu = instance.df_indiv[
    #         instance.df_indiv[it.tick_field] == t][it.metrics[0]].sum()
    #     max_t_mem = instance.df_indiv[
    #         instance.df_indiv[it.tick_field] == t]['mem'].sum()

    #     if max_t_cpu > max_cpu:
    #         max_cpu = max_t_cpu
    #     if max_t_mem > max_mem:
    #         max_mem = max_t_mem

    # # TODO consider nodes with different capacities
    # cap_cpu = instance.df_host_meta[it.metrics[0]].to_numpy()[0]
    # cap_mem = instance.df_host_meta['mem'].to_numpy()[0]
    # min_nodes_cpu = math.ceil(max_cpu / cap_cpu)
    # min_nodes_mem = math.ceil(max_mem / cap_mem)
    # return max(min_nodes_cpu, min_nodes_mem)

    max_metric = 0.0
    for t in range(total_time):
        max_t_metric = instance.df_indiv[
            instance.df_indiv[it.tick_field] == t][it.metrics[0]].sum()
        if max_t_metric > max_metric:
            max_metric = max_t_metric

    cap_metric = instance.df_host_meta[it.metrics[0]].to_numpy()[0]
    return (math.ceil(max_metric / cap_metric))


# TODO integrate upper bound for considering all clusters sum variance < ub
def place_opposite_clusters(
    instance, cluster_vars, cluster_var_matrix, labels_, min_nodes, conso_nodes
):
    """Initialize allocation heuristic by co-localizing distant clusters.

    :param instance: _description_
    :type instance: Instance
    :param cluster_vars: _description_
    :type cluster_vars: np.array
    :param cluster_var_matrix: _description_
    :type cluster_var_matrix: np.array
    :param labels_: _description_
    :type labels_: List
    :param min_nodes: _description_
    :type min_nodes: int
    :param conso_nodes: _description_
    :type conso_nodes: np.array
    :return: _description_
    :rtype: Tuple[np.array, np.array]
    """
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

    it_cont = colocalize_clusters(list_containers_i, list_containers_j,
                                  instance, total_time, min_nodes,
                                  conso_nodes)

    if not (len(list_containers_i) == len(list_containers_j)):
        # we have to place remaining containers

        if it_cont < len(list_containers_j):
            list_containers = list_containers_j[it_cont:]
            spread_containers(list_containers, instance, conso_nodes,
                              total_time, min_nodes)

        elif it_cont < len(list_containers_i):
            list_containers = list_containers_i[it_cont:]
            spread_containers(list_containers, instance, conso_nodes,
                              total_time, min_nodes)

    cluster_done[i] = 1
    cluster_done[j] = 1
    return conso_nodes, cluster_done


def move_list_containers(mvg_conts, instance, tmin, tmax, order='max'):
    """Move the list of containers to move.

    :param mvg_conts: _description_
    :type mvg_conts: List
    :param instance: _description_
    :type instance: Instance
    :param tmin: _description_
    :type tmin: int
    :param tmax: _description_
    :type tmax: int
    :param order: _description_, defaults to 'max'
    :type order: str, optional
    """
    # Remove all moving containers from nodes first
    old_ids = {}
    for mvg_cont in mvg_conts:
        old_ids[mvg_cont] = instance.df_indiv.loc[
            instance.df_indiv[it.indiv_field] == instance.dict_id_c[mvg_cont]
        ][it.host_field].to_numpy()[0]
        remove_container_node(old_ids[mvg_cont], instance.dict_id_c[mvg_cont], instance)
    # TODO developp smart method for replace containers (based on clustering)
    # Assign them to a new node
    mvg_conts_cons = {}
    for mvg_cont in mvg_conts:
        mvg_conts_cons[mvg_cont] = instance.df_indiv.loc[
            instance.df_indiv[it.indiv_field] == instance.dict_id_c[mvg_cont]
        ][it.metrics[0]].to_numpy()
    order_indivs = ()
    if order == 'max':
        order_indivs = ((max(cons), c) for c, cons in mvg_conts_cons.items())
    elif order == 'mean':
        order_indivs = ((sum(cons) / len(cons), c) for c, cons in mvg_conts_cons.items())
    for val, c in sorted(order_indivs, reverse=True):
        move_container(c, instance, tmin, tmax, old_ids[mvg_cont])


# TODO what to do if can't open another node
def move_container(mvg_cont, instance, tmin, tmax, old_id):
    """Move `mvg_cont` to another node.

    :param mvg_cont: _description_
    :type mvg_cont: int
    :param instance: _description_
    :type instance: Instance
    :param tmin: _description_
    :type tmin: int
    :param tmax: _description_
    :type tmax: int
    :param old_id: _description_
    :type old_id: str
    """
    print('Moving container :', instance.dict_id_c[mvg_cont])
    working_df_indiv = instance.df_indiv[
        (instance.
         df_indiv[it.tick_field] >= tmin) & (
            instance.df_indiv[it.tick_field] <= tmax)]
    nb_open_nodes = working_df_indiv[it.host_field].nunique()
    cons_c = working_df_indiv.loc[
        working_df_indiv[
            it.indiv_field] == instance.dict_id_c[mvg_cont]
    ][it.metrics[0]].to_numpy()
    n = working_df_indiv.loc[
        working_df_indiv[
            it.indiv_field] == instance.dict_id_c[mvg_cont]
    ][it.host_field].to_numpy()[0]
    cap_node = instance.df_host_meta.loc[
        instance.df_host_meta[it.host_field] == n
    ][it.metrics[0]].to_numpy()[0]
    # conso_nodes = np.zeros((working_df_indiv[it.host_field].nunique(), duration))
    nodes = working_df_indiv[it.host_field].unique()
    n_int = 0
    new_n = None
    min_var = float('inf')
    for node in nodes:
        node_data = instance.df_host.loc[
            (instance.df_host[it.tick_field] >= tmin) & (
                instance.df_host[it.tick_field] <= tmax) & (
                instance.df_host[it.host_field] == node
            )
        ].groupby(
            instance.df_host[it.tick_field]
        )[it.metrics[0]].sum().to_numpy()
        if (np.all(np.less((node_data + cons_c), cap_node))) and (
                (node_data + cons_c).var() < min_var):
            new_n = node
            min_var = (node_data + cons_c).var()
        n_int += 1
    if new_n is None:
        print('Impossible to move %s on another existing node.' % instance.dict_id_c[mvg_cont])
        print('We need to open a new node')
        nb_open_nodes += 1
        n_int += 1
        new_n = instance.dict_id_n[n_int]

    assign_container_node(
        new_n,
        instance.dict_id_c[mvg_cont],
        instance,
        remove=False)

    # n = working_df_indiv.loc[
    #     working_df_indiv[
    #         it.indiv_field] == instance.dict_id_c[mvg_cont]
    # ][it.host_field].to_numpy()[0]
    # print(working_df_indiv.loc[
    #     working_df_indiv[
    #         it.indiv_field] == instance.dict_id_c[mvg_cont]
    # ][it.host_field].to_numpy()[0])
    # input()
    # n_int = ([k for k, v in instance.
    #           dict_id_n.items() if v == n][0] + 1) % nb_open_nodes
    # print('He was on %s' % n)
    # done = False
    # n_count = 1
    # while not done:
    #     # TODO check n <= min_nodes or infeasibility
    #     if np.all(np.less((conso_nodes[n_int] + cons_c), cap_node)):
    #         assign_container_node(
    #             instance.dict_id_n[n_int],
    #             instance.dict_id_c[mvg_cont],
    #             instance)
    #         done = True
    #     else:
    #         n_count += 1
    #         if n_count > nb_open_nodes:
    #             print('We need to open a new node')
    #             nb_open_nodes += 1
    #             conso_nodes = np.append(
    #                 conso_nodes, [np.zeros(duration)], axis=0)
    #             n_int = nb_open_nodes - 1
    #         else:
    #             n_int = (n_int + 1) % nb_open_nodes
    #         cap_node = instance.df_host_meta.loc[
    #             instance.
    #             df_host_meta[it.host_field] == instance.dict_id_n[n_int]
    #         ][it.metrics[0]].to_numpy()[0]
    print('He can go on %s' % new_n)


def build_placement_adj_matrix(df_indiv, dict_id_c):
    """Build the adjacency matrix of placement.

    :param df_indiv: _description_
    :type df_indiv: pd.DataFrame
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    :return: _description_
    :rtype: np.array
    """
    nodes_ = [None] * df_indiv[it.indiv_field].nunique()
    for c in range(len(nodes_)):
        nodes_[c] = df_indiv.loc[
            df_indiv[it.indiv_field] == dict_id_c[c]][
                it.host_field].to_numpy()[0]
    v = np.zeros((len(nodes_), len(nodes_)))
    for (i, j) in combinations(range(len(nodes_)), 2):
        if nodes_[i] == nodes_[j]:
            v[i, j] = 1
            v[j, i] = 1
    return v
