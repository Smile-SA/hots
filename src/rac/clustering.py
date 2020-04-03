import multiprocessing as mp
from typing import Callable, Dict, List

import numpy as np
from numpy.linalg import multi_dot, norm

import pandas as pd

import scipy.cluster.hierarchy as hac
import multiprocessing as mp
import numpy as np
import pandas as pd
from numpy.linalg import multi_dot, norm
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.linalg.lapack import dsyevr
from scipy.linalg import fractional_matrix_power
from tqdm import tqdm

from .instance import Instance


# Functions definitions #
# TODO add mem cols
def matrix_line(args: (str, pd.DataFrame)) -> Dict:
    key, data = args
    line = {}
    for row in data.iterrows():
        line[int(row[1]['timestamp'])] = row[1]['cpu']
        line['container_id'] = key
    return line


def build_matrix_indiv_attr(df: pd.DataFrame) -> pd.DataFrame:
    print('Setup for clustering ...')
    list_args = list(df.groupby(df['container_id']))
    lines = []
    pool = mp.Pool(processes=(mp.cpu_count() - 1))
    for line in tqdm(pool.imap(matrix_line, list_args), total=len(list_args)):
        lines.append(line)

    pool.close()
    pool.join()
    df_return = pd.DataFrame(data=lines)
    df_return.fillna(0, inplace=True)
    df_return.set_index('container_id', inplace=True)
    return df_return


def build_similarity_matrix(df: pd.DataFrame) -> np.array:
    print(df)
    dists = pdist(df, 'euclidean')
    # dists = pdist(df.T)
    df_euclid = pd.DataFrame(squareform(
        dists), columns=df.index, index=df.index)
    # df_euclid = pd.DataFrame(1/(1+squareform(dists)), columns=df.index,
    # index=df.index)
    # print(df_euclid)
    sim_matrix = df_euclid.to_numpy()
    print(sim_matrix)
    return sim_matrix


# TODO add perso spectral + check error algo
def perform_clustering(data: pd.DataFrame, algo: str, k: int
                       ) -> Callable[[pd.DataFrame, int], List]:
    switcher = {'hierarchical': hierarchical_clustering,
                'kmeans': k_means,
                'spectral': spectral_clustering,
                'spectral_perso': perso_spectral_clustering}
    func = switcher.get(algo)
    return func(data, k)


def k_means(data: pd.DataFrame, k: int) -> List:
    return KMeans(n_clusters=k).fit(data).labels_


def p_dist(data: pd.DataFrame, metric: str = 'euclidean') -> np.array:
    return pairwise_distances(data, metric=metric)


def spectral_clustering(data: pd.DataFrame, k: int) -> List:
    return SpectralClustering(n_clusters=k).fit(data).labels_


def hierarchical_clustering(data: pd.DataFrame, k: int) -> List:
    Z_all = hac.linkage(data, method='ward', metric='euclidean')
    clusters_all = hac.fcluster(Z_all, k, criterion='distance')
    clusters_all -= 1
    return clusters_all


# TODO seems not working correctly => to fix
def perso_spectral_clustering(data: pd.DataFrame, k: int) -> np.array:
    W = build_similarity_matrix(data)
    D = np.diag(np.sum(W, axis=0))
    D_min1_2 = fractional_matrix_power(D, -1 / 2)
    A = multi_dot([D_min1_2, W, D_min1_2])

    print(D)
    print(A)

    # dsyevr parameters
    # borneinf = 1e-8
    drange = 'A'
    eigenvaluesA, eigenvectorsA, _ = dsyevr(A, range=drange)
    print('Valeurs propres de A :')
    print(eigenvaluesA)
    print('Vecteurs propres de A :')
    print(eigenvectorsA)
    idxA = eigenvaluesA.argsort()[::-1]

    # Get R first eigenvectors (R = k ?)
    # i = 0
    # while eigenvaluesA[idxA[i]] > borneinf:
    #     i += 1
    i = k
    idxA = idxA[:i]
    lambda_ = eigenvaluesA[idxA]
    U = eigenvectorsA[idxA]

    print('Valeurs propres de A (> borneinf) :')
    print(lambda_)

    print('Vecteurs propres U=(u1, ..., up) :')
    print(U)

    return(np.asarray(weighted_kmeans(W, D, U, k)))


def compute_mu_r(W: np.array, D: np.array, labels_: List, r: int, U: np.array
                 ) -> float:
    p = 0
    mu_r = 0
    d_p = 0
    for p in range(len(labels_)):
        if labels_[p] == r:
            mu_r += pow(D[p, p], 1 / 2) * U[:, p]
            d_p += D[p, p]

    return (mu_r * (1 / d_p))


def compute_distance_cluster_r(p: int, r: int, mu_r: float,
                               U: np.array, dp: float) -> float:
    # return math.sqrt(pow(
    # (U[:, p] * pow(dp, -1/2)) - mu_r, 2))
    return norm((U[:, p] * pow(dp, -1 / 2)) - mu_r)


def weighted_kmeans(W: np.array, D: np.array,
                    U: np.array, k: int) -> List:
    labels_ = [0] * len(W)
    n = 0
    i = 0
    while n < len(W):
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
            mu_[r] = compute_mu_r(W, D, labels_, r, U)
        for p in range(len(labels_)):
            dist_min = 1
            dist_r = 0
            r = 0
            for r in range(k):
                dist_Ar = compute_distance_cluster_r(p, r, mu_[r], U, D[p, p])
                if dist_Ar < dist_min:
                    dist_min = dist_Ar
                    dist_r = r
            if not labels_[p] == dist_r:
                labels_[p] = dist_r
                stationnary = False
        nb_loops += 1

    return labels_


def kMedoids(D: np.array, k: int, tmax: int = 100) -> (np.array, Dict):
    # determine dimensions of distance matrix D
    _, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs, cs = np.where(D == 0)  # pylint: disable=unbalanced-tuple-unpacking
    # the rows, cols must be shuffled because we will keep the first duplicate
    # below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r, c in zip(rs, cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception(
            'too many medoids (after removing {} duplicate points)'.format(
                len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for _ in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]

    # return results
    return M, C


def get_cluster_variance(nb_clusters: int, df_clust: pd.DataFrame) -> np.array:
    vars_ = np.zeros((nb_clusters), dtype=float)

    for key, data in df_clust.groupby(['cluster']):
        vars_[key] = data.drop('cluster', 1).sum().var()

    return vars_


def get_cluster_mean_profile(nb_clusters: int, df_clust: pd.DataFrame,
                             total_time: int, tmin: int = 0) -> np.array:
    profiles_ = np.zeros((nb_clusters, total_time), dtype=float)

    for key, data in df_clust.groupby(['cluster']):
        for t in range(total_time):
            profiles_[key, t] = data[t + tmin].mean()

    return profiles_


def get_sumCluster_variance(profiles_: np.array, vars_: np.array) -> np.array:
    k = len(profiles_)
    sumProfiles_matrix = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            sumProfiles_matrix[i, j] = (
                (profiles_[i, :] + profiles_[j, :]).var()) / (
                    vars_[i] + vars_[j])
            sumProfiles_matrix[j, i] = sumProfiles_matrix[i, j]

    return sumProfiles_matrix


def get_distanceCluster(instance: Instance, cluster_centers_: np.array
                        ) -> np.array:
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


def get_distance_containerCluster(conso_cont: np.array, profile: np.array
                                  ) -> float:
    """
    Compute the distance between the container profile and his cluster's
    mean profile.
    """
    return np.absolute(profile - conso_cont).mean() / profile.mean()


def check_container_deviation(
        working_df: pd.DataFrame, labels_: List, profiles_: np.array,
        dict_id_c: Dict) -> None:
    # print(profiles_)
    # print(labels_)

    for c in range(len(labels_)):
        dist = get_distance_containerCluster(
            working_df.loc[
                working_df['container_id'] == c
            ]['cpu'].to_numpy(), profiles_[labels_[c]])
        if dist > 0.5:
            print('Deviation of container ', c, dist)
