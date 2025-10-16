# hots/plugins/clustering/builder.py

"""Clustering builder utilities for HOTS."""

from itertools import combinations

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
