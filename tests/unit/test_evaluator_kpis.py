"""Unit tests for the evaluator KPI helpers."""

import networkx as nx
import numpy as np
import pandas as pd

from hots.evaluation.evaluator import (
    _entropy_from_counts,
    _gini,
    _kpi_clustering,
    _kpi_cluster_structure,
    _kpi_conflict_graph,
    _kpi_load_balance,
    _kpi_migrations_from_moves,
    _kpi_placement,
    _placement_from_working_df,
)


def test_kpi_clustering_without_previous_window():
    kpi = _kpi_clustering({"c_1": 0, "c_2": 1}, None)
    assert kpi["clusters_now"] == 2.0
    assert kpi["clusters_prev"] == 0.0
    assert kpi["reassigned_ratio"] == 0.0
    assert kpi["adjusted_rand_index"] == 0.0


def test_kpi_clustering_matches_containers_by_id():
    """A new container must not shift the comparison of the existing ones."""
    prev = {"c_1": 0, "c_2": 0, "c_3": 1}
    # c_0 is new and sorts first; the three known containers are unchanged
    now = {"c_0": 1, "c_1": 0, "c_2": 0, "c_3": 1}

    kpi = _kpi_clustering(now, prev)
    assert kpi["reassigned_ratio"] == 0.0
    assert kpi["adjusted_rand_index"] == 1.0


def test_kpi_clustering_counts_real_reassignments():
    prev = {"c_1": 0, "c_2": 0, "c_3": 1, "c_4": 1}
    now = {"c_1": 0, "c_2": 1, "c_3": 1, "c_4": 1}  # c_2 moved cluster

    kpi = _kpi_clustering(now, prev)
    assert kpi["reassigned_ratio"] == 0.25
    assert kpi["adjusted_rand_index"] < 1.0


def test_kpi_clustering_is_invariant_to_relabelling():
    """Swapping cluster names is not a clustering change: ARI must stay 1.0."""
    prev = {"c_1": 0, "c_2": 0, "c_3": 1}
    now = {"c_1": 1, "c_2": 1, "c_3": 0}

    kpi = _kpi_clustering(now, prev)
    assert kpi["adjusted_rand_index"] == 1.0
    assert kpi["reassigned_ratio"] == 1.0  # raw label mismatch, hence ARI


def test_kpi_placement_moved_ratio():
    prev = {"c_1": "n_1", "c_2": "n_1", "c_3": "n_2"}
    now = {"c_1": "n_2", "c_2": "n_1", "c_3": "n_2"}

    kpi = _kpi_placement(now, prev)
    assert kpi["assigned_now"] == 3.0
    assert kpi["assigned_prev"] == 3.0
    assert kpi["moved_ratio"] == 1 / 3


def test_kpi_placement_without_previous_window():
    kpi = _kpi_placement({"c_1": "n_1"}, None)
    assert kpi["moved_ratio"] == 0.0
    assert kpi["assigned_prev"] == 0.0


def test_kpi_cluster_structure():
    labels = np.array([0, 0, 0, 1, 2])  # sizes 3, 1, 1
    kpi = _kpi_cluster_structure(labels)

    assert kpi["n_clusters"] == 3.0
    assert kpi["size_min"] == 1.0
    assert kpi["size_max"] == 3.0
    assert kpi["singleton_ratio"] == 2 / 3
    assert kpi["size_entropy"] > 0.0


def test_kpi_cluster_structure_empty():
    kpi = _kpi_cluster_structure(np.array([]))
    assert kpi["n_clusters"] == 0.0
    assert kpi["size_gini"] == 0.0


def test_gini_and_entropy_edges():
    assert _gini(np.array([])) == 0.0
    assert _gini(np.array([0.0, 0.0])) == 0.0
    assert _gini(np.array([5.0, 5.0, 5.0])) == 0.0  # perfectly equal
    assert _gini(np.array([0.0, 0.0, 9.0])) > 0.5  # concentrated

    assert _entropy_from_counts(np.array([0.0])) == 0.0
    assert _entropy_from_counts(np.array([1.0, 1.0])) > 0.0


def test_kpi_conflict_graph_empty():
    kpi = _kpi_conflict_graph(nx.Graph())
    assert kpi["nodes"] == 0.0
    assert kpi["density"] == 0.0


def test_kpi_conflict_graph_weights_and_components():
    g = nx.Graph()
    g.add_edge("a", "b", weight=2.0)
    g.add_edge("c", "d", weight=4.0)

    kpi = _kpi_conflict_graph(g)
    assert kpi["nodes"] == 4.0
    assert kpi["edges"] == 2.0
    assert kpi["components"] == 2.0
    assert kpi["largest_component_ratio"] == 0.5
    assert kpi["w_sum"] == 6.0
    assert kpi["w_max"] == 4.0


def test_kpi_migrations_from_moves():
    assert _kpi_migrations_from_moves(None, n_indiv=4)["moves_count"] == 0.0
    kpi = _kpi_migrations_from_moves(["c_1", "c_2"], n_indiv=4)
    assert kpi["moves_count"] == 2.0
    assert kpi["moves_ratio"] == 0.5
    # no containers: ratio must not divide by zero
    assert _kpi_migrations_from_moves(["c_1"], n_indiv=0)["moves_ratio"] == 0.0


def test_kpi_load_balance():
    df = pd.DataFrame(
        {
            "timestamp": [0, 0, 1, 1],
            "container_id": ["c_1", "c_2", "c_1", "c_2"],
            "machine_id": ["n_1", "n_2", "n_1", "n_2"],
            "cpu": [1.0, 3.0, 1.0, 3.0],
        }
    )
    kpi = _kpi_load_balance(df, nf="container_id", hf="machine_id", metrics=["cpu"])

    assert kpi["n_hosts"] == 2.0
    assert kpi["n_indiv"] == 2.0
    assert kpi["host_load_mean"] == 4.0  # loads are 2.0 and 6.0
    assert kpi["host_load_max"] == 6.0
    assert kpi["host_load_cv"] > 0.0


def test_kpi_load_balance_empty_df():
    kpi = _kpi_load_balance(pd.DataFrame(), nf="container_id", hf="machine_id", metrics=["cpu"])
    assert kpi["n_hosts"] == 0.0
    assert kpi["host_load_gini"] == 0.0


def test_placement_from_working_df_uses_last_tick():
    df = pd.DataFrame(
        {
            "timestamp": [0, 0, 1, 1],
            "container_id": ["c_1", "c_2", "c_1", "c_2"],
            "machine_id": ["n_1", "n_2", "n_2", "n_2"],  # c_1 moved at tick 1
            "cpu": [1.0, 1.0, 1.0, 1.0],
        }
    )
    plc = _placement_from_working_df(df, nf="container_id", hf="machine_id", tf="timestamp")
    assert plc == {"c_1": "n_2", "c_2": "n_2"}

    plc_t0 = _placement_from_working_df(
        df, nf="container_id", hf="machine_id", tf="timestamp", tick=0
    )
    assert plc_t0 == {"c_1": "n_1", "c_2": "n_2"}


def test_placement_from_working_df_empty():
    assert _placement_from_working_df(pd.DataFrame(), nf="c", hf="h", tf="t") == {}
