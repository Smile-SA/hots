# hots/plugins/clustering/builder.py

"""Clustering builder utilities for HOTS."""

from itertools import combinations
import logging

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


def dist_from_mean(df_clust, profiles, cid: str) -> float:
    """Return distance from cid to its cluster mean profile."""
    row = df_clust.loc[cid]           # Series for that container
    k = int(row['cluster'])           # cluster id
    x = row.drop(labels='cluster').to_numpy(dtype=float)  # features only
    mu = np.asarray(profiles[k], dtype=float)             # cluster mean
    return np.linalg.norm(x - mu)


def get_far_container(c1, c2, df_clust: pd.DataFrame, profiles: np.ndarray) -> str:
    """Return c1 if it's farther from its cluster mean than c2 is, else return c2."""
    d1 = dist_from_mean(df_clust, profiles, c1)
    d2 = dist_from_mean(df_clust, profiles, c2)
    return c1 if d1 > d2 else c2


def change_clustering(
    mvg_containers,
    df_clust: pd.DataFrame,
    clustering,
    dict_id_c: dict,
    tol_open_clust: float = None
):
    """
    Reassign each container in mvg_containers to the closest existing cluster
    (by Euclidean distance to the cluster mean profile).
    """
    nb_changes = 0

    # Work only on moved-out base set to compute centroids (don’t include movers)
    df_clust_new = df_clust.loc[~df_clust.index.isin(mvg_containers)]
    if df_clust_new.empty:
        # No reference data to compute centroids -> nothing to do
        return df_clust, nb_changes

    profiles = cluster_mean_profile(df_clust_new)
    if profiles is None or (isinstance(profiles, np.ndarray) and profiles.size == 0):
        return df_clust, nb_changes

    # Columns that are features (everything except 'cluster')
    feature_cols = [c for c in df_clust.columns if c != 'cluster']

    for indiv in mvg_containers:
        # Skip if the container isn’t present
        if indiv not in df_clust.index:
            continue

        # Extract feature vector for this container
        row = df_clust.loc[indiv]
        x = row[feature_cols].to_numpy(dtype=float)

        # Distances to each cluster centroid
        # profiles[k] must be same length as x
        dists = np.linalg.norm(profiles - x, axis=1)
        new_cluster = int(np.argmin(dists))

        # --- Optional "open a new cluster" logic ---
        # min_dist = float(dists[new_cluster])
        # if tol_open_clust is not None and min_dist >= tol_open_clust:
        #     # create new cluster with this container as its centroid
        #     new_cluster = profiles.shape[0]
        #     profiles = np.vstack([profiles, x[None, :]])

        old_cluster = int(row['cluster'])
        if new_cluster != old_cluster:
            try:
                logging.info(
                    f'{indiv} changes cluster : from {old_cluster} to {new_cluster}\n'
                )
            except Exception:
                pass

            # Update df and labels_
            df_clust.loc[indiv, 'cluster'] = new_cluster
            nb_changes += 1

            # Update labels_ only if we can resolve the integer id
            c_int = dict_id_c.get(indiv, None)
            if c_int is not None and 0 <= c_int < len(clustering.labels):
                clustering.labels[c_int] = new_cluster

    return df_clust, nb_changes
