"""
Provide clustering algorithms and all clustering-related methods.
Here are the available clustering algorithms : k-means, hierarchical,
spectral, custom spectral.
"""

import multiprocessing as mp
from itertools import combinations

import networkx as nx

import numpy as np
from numpy.linalg import multi_dot, norm

import pandas as pd

import scipy.cluster.hierarchy as hac
from scipy.linalg import fractional_matrix_power
from scipy.linalg.lapack import dsyevr
from scipy.spatial.distance import pdist, squareform

from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances

from tqdm import tqdm

from . import init as it


# Functions definitions #
# TODO add mem cols
def matrix_line(args):
    """Build one line for clustering matrix.

    :param args: _description_
    :type args: Tuple[str, pd.DataFrame]
    :return: _description_
    :rtype: Tuple[int, Dict]
    """
    key, data = args
    line = {}
    for row in data.iterrows():
        line[int(row[1][it.tick_field])] = row[1][it.metrics[0]]
        line[it.indiv_field] = key
    return (key, line)


def build_matrix_indiv_attr(df):
    """Build entire clustering matrix.

    :param df: _description_
    :type df: pd.DataFrame
    :return: _description_
    :rtype: Tuple[pd.DataFrame, Dict]
    """
    print('Setup for clustering ...')
    list_args = list(df.groupby(df[it.indiv_field]))
    lines = []
    dict_id_c = {}
    int_id = 0
    if mp.cpu_count() > 0:
        pool = mp.Pool(processes=(mp.cpu_count() - 1))
    else:
        pool = mp.Pool(processes=(mp.cpu_count()))
    for (key, line) in tqdm(
            pool.imap(matrix_line, list_args), total=len(list_args)):
        lines.append(line)
        dict_id_c[int_id] = key
        int_id += 1

    pool.close()
    pool.join()
    df_return = pd.DataFrame(data=lines)
    df_return.fillna(0, inplace=True)
    df_return.set_index(it.indiv_field, inplace=True)
    return (df_return, dict_id_c)


def build_similarity_matrix(df):
    """Build a similarity matrix for the clustering.

    :param df: _description_
    :type df: pd.DataFrame
    :return: _description_
    :rtype: np.array
    """
    dists = pdist(df, 'euclidean')
    # dists = pdist(df.T)
    df_euclid = pd.DataFrame(squareform(
        dists), columns=df.index, index=df.index)
    # df_euclid = pd.DataFrame(1/(1+squareform(dists)), columns=df.index,
    # index=df.index)
    # print(df_euclid)
    sim_matrix = df_euclid.to_numpy()
    return sim_matrix


# TODO add perso spectral + check error algo
def perform_clustering(data, algo, k):
    """Call the specified method to perform clustering.

    :param data: _description_
    :type data: pd.DataFrame
    :param algo: _description_
    :type algo: str
    :param k: _description_
    :type k: int
    :return: _description_
    :rtype: Callable[[pd.DataFrame, int], List]
    """
    switcher = {'hierarchical': hierarchical_clustering,
                'kmeans': k_means,
                'spectral': spectral_clustering,
                'spectral_perso': perso_spectral_clustering}
    func = switcher.get(algo)
    return func(data, k)


def k_means(data, k):
    """Perform the K-means clustering.

    :param data: _description_
    :type data: pd.DataFrame
    :param k: _description_
    :type k: int
    :return: _description_
    :rtype: List
    """
    return KMeans(n_clusters=k, n_init='auto').fit(data).labels_


def p_dist(data, metric='euclidean'):
    """Compute distances between each data point pair.

    :param data: _description_
    :type data: pd.DataFrame
    :param metric: _description_, defaults to 'euclidean'
    :type metric: str, optional
    :return: _description_
    :rtype: np.array
    """
    return pairwise_distances(data, metric=metric)


def spectral_clustering(data, k):
    """Perform the spectral clustering.

    :param data: _description_
    :type data: pd.DataFrame
    :param k: _description_
    :type k: int
    :return: _description_
    :rtype: List
    """
    return SpectralClustering(n_clusters=k).fit(data).labels_


def hierarchical_clustering(data, k):
    """Perform the hierarchical ascendant clustering.

    :param data: _description_
    :type data: pd.DataFrame
    :param k: _description_
    :type k: int
    :return: _description_
    :rtype: List
    """
    z_all = hac.linkage(data, method='ward', metric='euclidean')
    clusters_all = hac.fcluster(z_all, k, criterion='distance')
    clusters_all -= 1
    return clusters_all


# TODO seems not working correctly => to fix
def perso_spectral_clustering(data, k):
    """Perform a customized version of spectral clustering.

    :param data: _description_
    :type data: pd.DataFrame
    :param k: _description_
    :type k: int
    :return: _description_
    :rtype: np.array
    """
    w = build_similarity_matrix(data)
    d = np.diag(np.sum(w, axis=0))
    d_min1_2 = fractional_matrix_power(d, -1 / 2)
    a = multi_dot([d_min1_2, w, d_min1_2])

    print(d)
    print(a)

    # dsyevr parameters
    # borneinf = 1e-8
    drange = 'a'
    eigenvalues_a, eigenvectors_a, _ = dsyevr(a, range=drange)
    print('Valeurs propres de a :')
    print(eigenvalues_a)
    print('Vecteurs propres de a :')
    print(eigenvectors_a)
    idx_a = eigenvalues_a.argsort()[::-1]

    # Get R first eigenvectors (R = k ?)
    # i = 0
    # while eigenvalues_a[idx_a[i]] > borneinf:
    #     i += 1
    i = k
    idx_a = idx_a[:i]
    lambda_ = eigenvalues_a[idx_a]
    u = eigenvectors_a[idx_a]

    print('Valeurs propres de A (> borneinf) :')
    print(lambda_)

    print('Vecteurs propres u=(u1, ..., up) :')
    print(u)

    return (np.asarray(weighted_kmeans(w, d, u, k)))


def compute_mu_r(w, d, labels_, r, u):
    """Compute center of cluster r.

    :param w: _description_
    :type w: np.array
    :param d: _description_
    :type d: np.array
    :param labels_: _description_
    :type labels_: List
    :param r: _description_
    :type r: int
    :param u: _description_
    :type u: np.array
    :return: _description_
    :rtype: float
    """
    p = 0
    mu_r = 0
    d_p = 0
    for p in range(len(labels_)):
        if labels_[p] == r:
            mu_r += pow(d[p, p], 1 / 2) * u[:, p]
            d_p += d[p, p]

    return (mu_r * (1 / d_p))


def compute_distance_cluster_r(p, r, mu_r, u, dp):
    """Compute distance between individual p and cluster r.

    :param p: _description_
    :type p: int
    :param r: _description_
    :type r: int
    :param mu_r: _description_
    :type mu_r: float
    :param u: _description_
    :type u: np.array
    :param dp: _description_
    :type dp: float
    :return: _description_
    :rtype: float
    """
    # return math.sqrt(pow(
    # (u[:, p] * pow(dp, -1/2)) - mu_r, 2))
    return norm((u[:, p] * pow(dp, -1 / 2)) - mu_r)


def weighted_kmeans(w, d, u, k):
    """Perform K-means algo for custom spectral clustering.

    :param w: _description_
    :type w: np.array
    :param d: _description_
    :type d: np.array
    :param u: _description_
    :type u: np.array
    :param k: _description_
    :type k: int
    :return: _description_
    :rtype: List
    """
    labels_ = [0] * len(w)
    n = 0
    i = 0
    while n < len(w):
        labels_[n] = i
        i = (i + 1) % k
        n += 1

    nb_loops = 1
    stationnary = False
    while not stationnary:
        print('Loop number ', nb_loops)
        stationnary = True
        r = 0
        mu_ = [0] * k
        for r in range(k):
            mu_[r] = compute_mu_r(w, d, labels_, r, u)
        for p in range(len(labels_)):
            dist_min = 1
            dist_r = 0
            r = 0
            for r in range(k):
                dist_ar = compute_distance_cluster_r(p, r, mu_[r], u, d[p, p])
                if dist_ar < dist_min:
                    dist_min = dist_ar
                    dist_r = r
            if not labels_[p] == dist_r:
                labels_[p] = dist_r
                stationnary = False
        nb_loops += 1

    return labels_


def get_cluster_variance(profiles_):
    """Compute the variance of each cluster.

    :param profiles_: _description_
    :type profiles_: np.array
    :return: _description_
    :rtype: np.array
    """
    k = len(profiles_)
    vars_ = np.zeros((k), dtype=float)
    for i in range(k):
        vars_[i] = profiles_[i].var()
    return vars_


def get_cluster_mean_profile(df_clust):
    """Compute the mean profile of each cluster.

    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :return: _description_
    :rtype: np.array
    """
    profiles_ = np.zeros((
        df_clust['cluster'].nunique(),
        len(df_clust.columns) - 1), dtype=float)
    for key, data in df_clust.groupby('cluster'):
        t = 0
        for c, c_data in data.iloc[:, :-1].items():
            profiles_[key, t] = c_data.mean()
            t += 1
    return profiles_


def get_sum_cluster_variance(profiles_, vars_):
    """Compute a matrix of sum of variances of each pair of cluster.

    :param profiles_: _description_
    :type profiles_: np.array
    :param vars_: _description_
    :type vars_: np.array
    :return: _description_
    :rtype: np.array
    """
    k = len(profiles_)
    sum_profiles_matrix = np.full((k, k), -1, dtype=float)
    # for i in range(k):
    #     for j in range(i + 1, k):
    #         if vars_[i] + vars_[j] == 0.0:
    #             sum_profiles_matrix[i, j] = 1.0
    #         else:
    #             sum_profiles_matrix[i, j] = (
    #                 (profiles_[i, :] + profiles_[j, :]).var(ddof=0)) / (
    #                 vars_[i] + vars_[j])
    #         sum_profiles_matrix[j, i] = sum_profiles_matrix[i, j]
    #     sum_profiles_matrix[i, i] = -1.0
    # print(sum_profiles_matrix)
    # sum_profiles_matrix = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            sum_profiles_matrix[i, j] = (
                (profiles_[i, :] + profiles_[j, :]).var(ddof=0))
            sum_profiles_matrix[j, i] = sum_profiles_matrix[i, j]
    return sum_profiles_matrix


def get_distance_cluster(instance, cluster_centers_):
    """Compute the distance between each cluster.

    :param instance: _description_
    :type instance: Instance
    :param cluster_centers_: _description_
    :type cluster_centers_: np.array
    :return: _description_
    :rtype: np.array
    """
    print('Compute distance between each cluster ...')
    cluster_distance = np.zeros(
        (instance.nb_clusters, instance.nb_clusters), dtype=float)

    pbar = tqdm(range(instance.nb_clusters))
    for row1 in pbar:
        for row2 in range(row1 + 1, instance.nb_clusters):
            cluster_distance[row1][row2] = np.linalg.norm(
                cluster_centers_[row1] - cluster_centers_[row2])
            cluster_distance[row2][row1] = cluster_distance[row1][row2]
    return cluster_distance


def get_distance_container_cluster(conso_cont, profile):
    """
    Compute the distance between the container profile and his cluster's
    mean profile.

    :param conso_cont: _description_
    :type conso_cont: np.array
    :param profile: _description_
    :type profile: np.array
    :return: _description_
    :rtype: float
    """
    return np.absolute(profile - conso_cont).mean() / profile.mean()


def check_container_deviation(
    working_df, labels_, profiles_, dict_id_c
):
    """Check the deviation of container from its cluster.

    :param working_df: _description_
    :type working_df: pd.DataFrame
    :param labels_: _description_
    :type labels_: List
    :param profiles_: _description_
    :type profiles_: np.array
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    """
    for c in range(len(labels_)):
        dist = get_distance_container_cluster(
            working_df.loc[
                working_df[it.indiv_field] == dict_id_c[c]
            ]['cpu'].to_numpy(), profiles_[labels_[c]])
        if dist > 0.5:
            print('Deviation of container ', c, dist)


def build_adjacency_matrix(labels_):
    """Build the adjacency matrix of clustering.

    :param labels_: _description_
    :type labels_: _type_
    :return: _description_
    :rtype: np.array
    """
    u = np.zeros((len(labels_), len(labels_)))
    for (i, j) in combinations(range(len(labels_)), 2):
        if labels_[i] == labels_[j]:
            u[i, j] = 1
            u[j, i] = 1
    return u


def get_cluster_balance(df_clust):
    """Display size of each cluster.

    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    """
    print('Clustering balance : ',
          df_clust.groupby('cluster').count())


def get_far_container(c1, c2, df_clust, profiles):
    """Get the farthest container between c1 and c2 compared to profile.

    :param c1: _description_
    :type c1: str
    :param c2: _description_
    :type c2: str
    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :param profiles: _description_
    :type profiles: np.array
    :return: _description_
    :rtype: str
    """
    if norm(
        df_clust.loc[c1].drop('cluster').values
        - profiles[int(df_clust.loc[c1]['cluster'])]) >= norm(
            df_clust.loc[c2].drop('cluster').values
            - profiles[int(df_clust.loc[c2]['cluster'])]):
        return c1
    else:
        return c2


def change_clustering(mvg_containers, df_clust, labels_, dict_id_c, tol_open_clust):
    """Adjust the clustering with individuals to move to the closest cluster.

    :param mvg_containers: _description_
    :type mvg_containers: List
    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :param labels_: _description_
    :type labels_: List
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    :param tol_open_clust: _description_
    :type tol_open_clust: float
    :return: _description_
    :rtype: Tuple[pd.DataFrame, List, int]
    """
    nb_changes = 0
    df_clust_new = df_clust[~df_clust.index.isin(mvg_containers)]
    profiles = get_cluster_mean_profile(df_clust_new)
    for indiv in mvg_containers:
        # print('Dealing with indiv ', indiv)
        min_dist = float('inf')
        new_cluster = -1
        for cluster in range(len(profiles)):
            if norm(
                df_clust.loc[indiv].drop('cluster').values - profiles[cluster]
            ) < min_dist:
                min_dist = norm(
                    df_clust.loc[indiv].drop('cluster').values - profiles[cluster]
                )
                new_cluster = cluster
        # TODO not ideal for the moment : consider dynamic threshold for conflict graph
        # if min_dist >= tol_open_clust:
        #     print('We open a new cluster')
        #     new_cluster = cluster + 1
        #     profiles = np.append(
        #         profiles,
        #         [df_clust.loc[indiv].drop('cluster').values],
        #         axis=0)
        if new_cluster != df_clust.loc[indiv, 'cluster']:
            it.results_file.write('%s changes cluster : from %d to %d\n' % (
                indiv, df_clust.loc[indiv, 'cluster'], new_cluster))
            print('%s changes cluster : from %d to %d\n' % (
                indiv, df_clust.loc[indiv, 'cluster'], new_cluster))
            df_clust.loc[indiv, 'cluster'] = new_cluster
            nb_changes += 1
            c_int = [k for k, v in dict_id_c.items() if v == indiv][0]
            labels_[c_int] = new_cluster
    # print('New clustering : ', labels_)
    return (df_clust, labels_, nb_changes)


def change_clustering_maxkcut(conflict_graph, df_clust, labels_, dict_id_c):
    """Change current clustering with max-k-cut on moving containers.

    :param conflict_graph: _description_
    :type conflict_graph: nx.Graph
    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :param labels_: _description_
    :type labels_: List
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    :return: _description_
    :rtype: Tuple[pd.DataFrame, List, int]
    """
    nb_changes = 0
    df_clust_new = df_clust
    labels_new = labels_
    # print('List violated constraints :')
    # print(constraints_kept)
    # print('List of indivs :')
    # print(mvg_containers)

    # df_clust_new = df_clust[~df_clust.index.isin(mvg_containers)]
    # print(df_clust_new)
    # print('nb pts : ', len(df_clust_new.columns) - 1)
    # labels_new = [None] * len(labels_)
    # for indiv in mvg_containers:
    #     print(indiv)
    # set overall obj
    # build list of clusters with indivs already in it ?
    # loop on indivs
    # get cluster of indiv and all neighbours
    # assign indiv to cluster that max overall obj

    mvg_containers = sorted(conflict_graph.degree, key=lambda x: x[1], reverse=True)
    df_clust_new = df_clust[~df_clust.index.isin(mvg_containers)]
    profiles = get_cluster_mean_profile(
        df_clust_new['cluster'].nunique(), df_clust_new
    )
    print(df_clust)
    print(df_clust_new)
    print(profiles)
    print(conflict_graph)
    print(mvg_containers)
    for indiv, occur in mvg_containers:
        indiv_s = dict_id_c[int(indiv)]
        min_dist = float('inf')
        new_cluster = -1
        print('Moving indiv ', indiv_s, occur)
        print(conflict_graph.edges.data())
        for cluster in range(len(profiles)):
            if norm(
                df_clust_new.loc[indiv_s].drop('cluster').values - profiles[cluster]
            ) < min_dist:
                min_dist = norm(
                    df_clust_new.loc[indiv_s].drop('cluster').values - profiles[cluster]
                )
                new_cluster = cluster
        if new_cluster != df_clust_new.loc[indiv_s, 'cluster']:
            it.results_file.write('%s changes cluster : from %d to %d\n' % (
                indiv_s, df_clust_new.loc[indiv_s, 'cluster'], new_cluster))
            df_clust_new.loc[indiv_s, 'cluster'] = new_cluster
            nb_changes += 1
            labels_new[indiv] = new_cluster
        else:
            print('indiv did not change cluster')
        conflict_graph.remove_node(indiv)
        print(conflict_graph.edges.data())
        print(list(nx.isolates(conflict_graph)))
        print(conflict_graph.remove_nodes_from(list(nx.isolates(conflict_graph))))
        print(sorted(conflict_graph.degree, key=lambda x: x[1], reverse=True))

        input()

    # return (df_clust_new, labels_new, nb_changes)


def eval_clustering(df_clust, w, dict_id_c):
    """Evaluate the clustering with ICS and ICD.

    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :param w: _description_
    :type w: np.array
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    :return: _description_
    :rtype: Tuple[float, float]
    """
    ics = 0.0
    icd = 0.0
    for (c1_s, c2_s) in combinations(list(df_clust.index.values), 2):
        c1 = [k for k, v in dict_id_c.items() if v == c1_s][0]
        c2 = [k for k, v in dict_id_c.items() if v == c2_s][0]
        if df_clust.loc[c1_s]['cluster'] == df_clust.loc[c2_s]['cluster']:
            ics += w[c1][c2]
        else:
            icd += w[c1][c2]
    return (ics, icd)


def get_silhouette(df_clust, labels_):
    """Get the Silhouette score from clustering.

    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :param labels_: _description_
    :type labels_: List
    :return: _description_
    :rtype: float
    """
    return metrics.silhouette_score(
        df_clust.drop('cluster', axis=1), labels_, metric='euclidean'
    )
