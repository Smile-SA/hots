"""Unit tests for clustering builder utilities."""

from hots.plugins.clustering.builder import (
    build_adjacency_matrix,
    build_matrix_indiv_attr,
    build_similarity_matrix,
    cluster_mean_profile,
    dist_from_mean,
    get_far_container,
    pairwise_sum_profile_var,
)

import numpy as np

import pandas as pd


def test_build_matrix_indiv_attr_basic():
    """
    build_matrix_indiv_attr should:

    - produce a container × tick matrix,
    - fill missing ticks with 0,
    - respect id_map ordering for the index.
    """
    df = pd.DataFrame(
        [
            {'cid': 'a', 'tick': 1, 'cpu': 1.0},
            {'cid': 'a', 'tick': 2, 'cpu': 2.0},
            {'cid': 'b', 'tick': 1, 'cpu': 0.5},
        ]
    )
    id_map = {'a': 0, 'b': 1}

    mat = build_matrix_indiv_attr(
        df=df,
        tick_field='tick',
        indiv_field='cid',
        metrics=['cpu'],
        id_map=id_map,
    )

    assert list(mat.index) == ['a', 'b']
    # Known values
    assert mat.loc['a', 1] == 1.0
    assert mat.loc['a', 2] == 2.0
    # Missing tick 2 for 'b' should be filled with 0
    assert mat.loc['b', 2] == 0.0


def test_build_adjacency_matrix_simple():
    """Check adjacency between same-label points only."""
    labels = [0, 0, 1]
    u = build_adjacency_matrix(labels)

    assert u.shape == (3, 3)
    # Same cluster -> 1
    assert u[0, 1] == u[1, 0] == 1
    # Different cluster -> 0
    assert u[0, 2] == u[2, 0] == 0
    assert np.all(np.diag(u) == 0)


def test_build_similarity_matrix_is_symmetric_and_zero_diag():
    """Distance matrix should be symmetric with zeros on the diagonal."""
    mat = pd.DataFrame([[0.0, 1.0], [1.0, 2.0]], index=['a', 'b'])
    sim = build_similarity_matrix(mat)

    assert sim.shape == (2, 2)
    assert np.allclose(np.diag(sim), 0.0)
    assert np.allclose(sim, sim.T)


def test_cluster_mean_profile_and_pairwise_var():
    """
    cluster_mean_profile should produce one mean vector per cluster,
    and pairwise_sum_profile_var should:
    - return a (k, k) matrix for k clusters.
    (We don't assert exact values, only shapes/invariants.)
    """
    # Only numeric features + cluster label.
    df_clust = pd.DataFrame(
        [
            {'cluster': 0, 'f1': 0.0, 'f2': 2.0},
            {'cluster': 0, 'f1': 2.0, 'f2': 4.0},
            {'cluster': 1, 'f1': 1.0, 'f2': 1.0},
        ]
    )

    profiles = cluster_mean_profile(df_clust)
    # Expect 2 clusters, 2 features
    assert profiles.shape == (2, 2)

    var_mat = pairwise_sum_profile_var(profiles)
    # Square matrix, same number of clusters
    assert var_mat.shape == (2, 2)


def test_dist_and_far_container():
    """
    dist_from_mean should give larger distance for containers further from the
    cluster mean; get_far_container should return the farther one.
    """
    df_clust = pd.DataFrame(
        [
            {'cid': 'c1', 'cluster': 0, 'f1': 1.0, 'f2': 1.0},
            {'cid': 'c2', 'cluster': 0, 'f1': 3.0, 'f2': 4.0},
        ]
    ).set_index('cid')

    profiles = np.array([[0.0, 0.0]])  # mean profile for cluster 0

    d1 = dist_from_mean(df_clust, profiles, 'c1')
    d2 = dist_from_mean(df_clust, profiles, 'c2')
    assert d2 > d1

    far = get_far_container('c1', 'c2', df_clust, profiles)
    assert far == 'c2'
