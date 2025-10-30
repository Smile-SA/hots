# hots/plugins/clustering/builder.py

"""Clustering builder utilities for HOTS."""

from itertools import combinations

from hots.utils.tools import check_missing_entries_df
import numpy as np

import pandas as pd

from scipy.spatial.distance import pdist, squareform


def build_matrix_indiv_attr(
    df: pd.DataFrame,
    tick_field: str,
    indiv_field: str,
    metrics: list,
    id_map: dict
) -> pd.DataFrame:
    """Build a container×time matrix from individual‐level DataFrame."""
    rows = []
    for cid, group in df.groupby(indiv_field):
        row = dict(zip(group[tick_field], group[metrics[0]]))
        row[indiv_field] = cid
        rows.append(row)
    mat = pd.DataFrame(rows).fillna(0).set_index(indiv_field)
    sorted_idx = sorted(mat.index, key=lambda x: id_map[x])
    print(df)
    print(mat.loc[sorted_idx])
    return mat.loc[sorted_idx]


def build_adjacency_matrix(labels_):
    """Build the adjacency matrix of clustering.

    :param labels_: List of clusters assigned to individuals
    :type labels_: List
    :return: Adjacency matrix
    :rtype: np.array
    """
    u = np.zeros((len(labels_), len(labels_)))
    for (i, j) in combinations(range(len(labels_)), 2):
        if labels_[i] == labels_[j]:
            u[i, j] = 1
            u[j, i] = 1
    return u


def build_similarity_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Euclidean distance matrix from input matrix."""
    return squareform(pdist(mat.values, 'euclidean'))


def build_pre_clust_matrices(
    df,
    tick_field,
    indiv_field,
    metrics,
    id_map,
    labels
):
    """Build period clustering dataframes and matrices to be used."""
    clust_mat = build_matrix_indiv_attr(
        df,
        tick_field,
        indiv_field,
        metrics,
        id_map
    )
    u_mat = build_adjacency_matrix(labels)
    w_mat = build_similarity_matrix(clust_mat)

    inv = {v: k for k, v in id_map.items()}
    idx = [inv[i] for i in range(len(labels))]
    labels_s = pd.Series(labels, index=idx, name='cluster')
    clust_mat = clust_mat.join(labels_s, how='left')

    return (
        clust_mat, u_mat, w_mat
    )


def build_post_clust_matrices(clust_mat):
    """Build result clustering dataframes and matrices to be used."""
    cluster_profiles = cluster_mean_profile(
        clust_mat)
    cluster_var_matrix = pairwise_sum_profile_var(
        cluster_profiles)
    dv_mat = build_var_delta_cluster_matrix(clust_mat, cluster_var_matrix)
    return dv_mat


def cluster_mean_profile(df_clust: pd.DataFrame, cluster_col: str = 'cluster') -> np.ndarray:
    """Compute the mean profile of each cluster."""
    # pick only numeric feature columns
    feature_cols = df_clust.columns.drop(cluster_col)
    # compute means per cluster
    grouped = df_clust.groupby(cluster_col)[feature_cols].mean()
    # if cluster labels are not 0..K-1, reindex to dense 0..K-1
    dense = grouped.reset_index(drop=True)
    return dense.to_numpy(dtype=float)


def pairwise_sum_profile_var(profiles: np.ndarray) -> np.ndarray:
    """Compute a matrix of variance of sum of profiles for each pair of cluster."""
    # profiles: (k, p)
    k, p = profiles.shape
    # expand to (k, 1, p) and (1, k, p) then broadcast-sum -> (k, k, p)
    summed = profiles[:, None, :] + profiles[None, :, :]  # (k, k, p)
    # variance along the feature axis
    var_mat = summed.var(axis=2, ddof=0)  # (k, k)
    np.fill_diagonal(var_mat, -1.0)
    return var_mat


def build_var_delta_cluster_matrix(df_clust, cluster_var_matrix, *, zero_diag=True):
    """Build variance of deltas matrix from cluster."""
    # labels[i] = cluster id for the i-th row in df_clust (must be ints in [0..K-1])
    labels = df_clust['cluster'].to_numpy()
    # Broadcast-select the (cluster_i, cluster_j) entry for all pairs
    vars_matrix = cluster_var_matrix[np.ix_(labels, labels)].astype(float, copy=False)
    if zero_diag:
        np.fill_diagonal(vars_matrix, 0.0)
    return vars_matrix
