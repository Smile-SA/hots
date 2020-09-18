"""
=========
cots clustering
=========

Provide clustering algorithms and all clustering-related methods.
Here are the available clustering algorithms : k-means, hierarchical,
spectral, custom spectral.
"""

import multiprocessing as mp
from typing import Callable, Dict, List

import numpy as np
from numpy.linalg import multi_dot, norm

import pandas as pd

import scipy.cluster.hierarchy as hac
from scipy.linalg import fractional_matrix_power
from scipy.linalg.lapack import dsyevr
from scipy.spatial.distance import pdist, squareform

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances

from tqdm import tqdm

from .instance import Instance


# Functions definitions #
# TODO add mem cols
def matrix_line(args: (str, pd.DataFrame)) -> (int, Dict):
    """Build one line for clustering matrix."""
    key, data = args
    line = {}
    for row in data.iterrows():
        line[int(row[1]['timestamp'])] = row[1]['cpu']
        line['container_id'] = key
    return (key, line)


def build_matrix_indiv_attr(df: pd.DataFrame) -> (pd.DataFrame, Dict):
    """Build entire clustering matrix."""
    print('Setup for clustering ...')
    list_args = list(df.groupby(df['container_id']))
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
    df_return.set_index('container_id', inplace=True)
    return (df_return, dict_id_c)


def build_similarity_matrix(df: pd.DataFrame) -> np.array:
    """Build a similarity matrix for the clustering."""
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
def perform_clustering(data: pd.DataFrame, algo: str, k: int
                       ) -> Callable[[pd.DataFrame, int], List]:
    """Call the specified method to perform clustering."""
    switcher = {'hierarchical': hierarchical_clustering,
                'kmeans': k_means,
                'spectral': spectral_clustering,
                'spectral_perso': perso_spectral_clustering}
    func = switcher.get(algo)
    return func(data, k)


def k_means(data: pd.DataFrame, k: int) -> List:
    """Perform the K-means clustering."""
    return KMeans(n_clusters=k).fit(data).labels_


def p_dist(data: pd.DataFrame, metric: str = 'euclidean') -> np.array:
    """Compute distances between each data point pair."""
    return pairwise_distances(data, metric=metric)


def spectral_clustering(data: pd.DataFrame, k: int) -> List:
    """Perform the spectral clustering."""
    return SpectralClustering(n_clusters=k).fit(data).labels_


def hierarchical_clustering(data: pd.DataFrame, k: int) -> List:
    """Perform the hierarchical ascendant clustering."""
    z_all = hac.linkage(data, method='ward', metric='euclidean')
    clusters_all = hac.fcluster(z_all, k, criterion='distance')
    clusters_all -= 1
    return clusters_all


# TODO seems not working correctly => to fix
def perso_spectral_clustering(data: pd.DataFrame, k: int) -> np.array:
    """Perform a customized version of spectral clustering."""
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

    return(np.asarray(weighted_kmeans(w, d, u, k)))


def compute_mu_r(w: np.array, d: np.array, labels_: List, r: int, u: np.array
                 ) -> float:
    """Compute center of cluster r."""
    p = 0
    mu_r = 0
    d_p = 0
    for p in range(len(labels_)):
        if labels_[p] == r:
            mu_r += pow(d[p, p], 1 / 2) * u[:, p]
            d_p += d[p, p]

    return (mu_r * (1 / d_p))


def compute_distance_cluster_r(p: int, r: int, mu_r: float,
                               u: np.array, dp: float) -> float:
    """Compute distance between individual p and cluster r."""
    # return math.sqrt(pow(
    # (u[:, p] * pow(dp, -1/2)) - mu_r, 2))
    return norm((u[:, p] * pow(dp, -1 / 2)) - mu_r)


def weighted_kmeans(w: np.array, d: np.array,
                    u: np.array, k: int) -> List:
    """Perform K-means algo for custom spectral clustering."""
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


def get_cluster_variance(nb_clusters: int, df_clust: pd.DataFrame) -> np.array:
    """Compute the variance of each cluster."""
    vars_ = np.zeros((nb_clusters), dtype=float)

    for key, data in df_clust.groupby(['cluster']):
        vars_[key] = data.drop('cluster', 1).sum().var()

    return vars_


def get_cluster_mean_profile(nb_clusters: int, df_clust: pd.DataFrame,
                             total_time: int, tmin: int = 0) -> np.array:
    """Compute the mean profile of each cluster."""
    profiles_ = np.zeros((nb_clusters, total_time), dtype=float)
    for key, data in df_clust.groupby(['cluster']):
        for t in range(total_time):
            profiles_[key, t] = data[t + tmin].mean()

    return profiles_


def get_sum_cluster_variance(profiles_: np.array, vars_: np.array) -> np.array:
    """Compute a matrix of sum of variances of each pair of cluster."""
    k = len(profiles_)
    sum_profiles_matrix = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            sum_profiles_matrix[i, j] = (
                (profiles_[i, :] + profiles_[j, :]).var()) / (
                    vars_[i] + vars_[j])
            sum_profiles_matrix[j, i] = sum_profiles_matrix[i, j]
        sum_profiles_matrix[i, i] = -1.0
    return sum_profiles_matrix


def get_distance_cluster(instance: Instance, cluster_centers_: np.array
                         ) -> np.array:
    """Compute the distance between each cluster."""
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


def get_distance_container_cluster(conso_cont: np.array, profile: np.array
                                   ) -> float:
    """
    Compute the distance between the container profile and his cluster's
    mean profile.
    """
    return np.absolute(profile - conso_cont).mean() / profile.mean()


def check_container_deviation(
        working_df: pd.DataFrame, labels_: List, profiles_: np.array,
        dict_id_c: Dict) -> None:
    """Check the deviation of container from its cluster."""
    # print(profiles_)
    # print(labels_)

    for c in range(len(labels_)):
        dist = get_distance_container_cluster(
            working_df.loc[
                working_df['container_id'] == dict_id_c[c]
            ]['cpu'].to_numpy(), profiles_[labels_[c]])
        if dist > 0.5:
            print('Deviation of container ', c, dist)