"""Evaluation utilities for HOTS."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from hots.plugins.clustering.builder import (
    build_adjacency_matrix,
    build_matrix_indiv_attr,
    build_similarity_matrix
)

import networkx as nx

import numpy as np

import pandas as pd

from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class EvalSnapshot:
    """All the info we need to compare two consecutive evaluations."""

    labels: np.ndarray                 # shape (n_indiv,)
    placement: Dict[Any, Any]          # mapping indiv/container -> host/node
    tick: int                          # current time tick for traceability


def _safe_labels_array(labels) -> np.ndarray:
    """Ensure labels are a 1D numpy array."""
    if isinstance(labels, np.ndarray):
        return labels
    # pandas Series or list-like
    return np.asarray(labels).reshape(-1)


def _kpi_clustering(labels_now: np.ndarray, labels_prev: Optional[np.ndarray]) -> Dict[str, float]:
    """Compare clustering KPIs + delta vs previous."""
    kpi: Dict[str, float] = {
        'clusters_now': float(len(np.unique(labels_now))),
    }
    if labels_prev is None:
        kpi.update({
            'clusters_prev': 0.0,
            'clustered_change_ratio': 0.0,
            'reassigned_ratio': 0.0,
        })
        return kpi

    labels_prev = _safe_labels_array(labels_prev)
    same_len = min(len(labels_prev), len(labels_now))
    reassigned = np.sum(labels_prev[:same_len] != labels_now[:same_len])
    kpi.update({
        'clusters_prev': float(len(np.unique(labels_prev))),
        'clustered_change_ratio': float(
            abs(len(np.unique(labels_now)) - len(np.unique(labels_prev)))
        ),
        'reassigned_ratio': float(reassigned) / float(same_len) if same_len else 0.0,
    })
    return kpi


def _kpi_placement(
    placement_now: Dict[Any, Any],
    placement_prev: Optional[Dict[Any, Any]],
) -> Dict[str, float]:
    """Placement KPIs + delta vs previous (e.g., % moved)."""
    moved_ratio = 0.0
    if placement_prev:
        common = set(placement_now.keys()) & set(placement_prev.keys())
        moved = sum(1 for k in common if placement_now[k] != placement_prev[k])
        moved_ratio = float(moved) / float(len(common)) if common else 0.0

    return {
        'assigned_now': float(len(placement_now)),
        'assigned_prev': float(len(placement_prev) if placement_prev else 0),
        'moved_ratio': moved_ratio,
    }


def eval_solutions(
    df_indiv: pd.DataFrame,
    df_host: pd.DataFrame,
    labels,
    clustering,
    optimization,
    problem,
    instance
) -> Tuple[Any, Dict[str, Any]]:
    """Run the evaluation pipeline and collect evaluation metrics."""
    # 1) Build & solve clustering problem
    # TODO update and no build
    mat = build_matrix_indiv_attr(
        df_indiv,
        instance.config.tick_field,
        instance.config.individual_field,
        instance.config.metrics,
        instance.get_id_map()
    )
    u_mat = build_adjacency_matrix(labels)
    w_mat = build_similarity_matrix(mat)
    sil = silhouette_score(mat.values, labels)

    # 2) Update & Solve the optimization model
    model = optimization.build(u_mat, w_mat)
    # model = optimization.solve(df_host, labels)
    model.solve(
        solver=instance.config.optimization.parameters.get('solver', 'glpk'),
    )

    # 3) Extract dual values
    duals = model.fill_dual_values()

    # 4) Read tolerances
    tol = instance.config.problem.parameters.get('tol', 0.1)
    tol_move = instance.config.problem.parameters.get('tol_move', 0.1)

    # 5) Conflict detection & pick moving containers
    pb = instance.config.optimization.parameters.get('pb_number', 1)
    if pb == 1:
        moving, nodes, edges, max_deg, mean_deg = get_moving_containers_clust(
            model,
            duals,
            tol,
            tol_move,
            df_clust=df_indiv,
            profiles=None,
        )
    else:
        moving, nodes, edges, max_deg, mean_deg = get_moving_containers_place(
            model,
            duals,
            tol,
            tol_move,
            working_df=df_indiv,
        )

    # 6) Apply business‑problem logic
    solution2 = problem.adjust(model, moving)

    # 7) Collect metrics
    metrics: Dict[str, Any] = {
        'silhouette': sil,
        'conflict_nodes': nodes,
        'conflict_edges': edges,
        'max_conf_degree': max_deg,
        'mean_conf_degree': mean_deg,
        'moving_containers': moving
    }

    return solution2, metrics


def get_conflict_graph(
    model,
    prev_duals: Dict[Any, float],
    tol: float
) -> nx.Graph:
    """Build conflict graph where edges represent dual increases above tolerance."""
    inst = model.instance_model
    dual = inst.dual

    if model.pb_number == 1:
        must_link = getattr(inst, 'must_link_c', {})
    else:
        must_link = getattr(inst, 'must_link_n', {})

    g = nx.Graph()
    for idx_pair, con in must_link.items():
        prev = prev_duals.get(idx_pair, 0.0)
        if prev <= 0:
            continue
        curr = dual[con]
        if curr > prev * (1 + tol):
            g.add_edge(idx_pair[0], idx_pair[1])
    return g


def get_moving_containers_clust(
    model,
    prev_duals: Dict[Any, float],
    tol: float,
    tol_move: float,
    df_clust: pd.DataFrame,
    profiles: np.ndarray
) -> Tuple[List[Any], int, int, int, float]:
    """Select containers to move from clustering conflict graph."""
    g = get_conflict_graph(model, prev_duals, tol)
    n_nodes, n_edges = g.number_of_nodes(), g.number_of_edges()
    degrees = sorted(g.degree(), key=lambda x: x[1], reverse=True)
    if not degrees:
        return [], n_nodes, n_edges, 0, 0.0
    max_deg = degrees[0][1]
    mean_deg = sum(d for _, d in degrees) / len(degrees)

    moving = []
    budget = len(model.dict_id_c) * tol_move
    while degrees and len(moving) < budget:
        cid, deg = degrees[0]
        if deg > 1:
            moving.append(cid)
            g.remove_node(cid)
        else:
            partner = next(iter(g.neighbors(cid)))
            to_move = get_far_container(model, cid, partner, df_clust, profiles)
            moving.append(to_move)
            g.remove_node(cid)
            if partner in g:
                g.remove_node(partner)
        g.remove_nodes_from(nx.isolates(g))
        degrees = sorted(g.degree(), key=lambda x: x[1], reverse=True)

    return moving, n_nodes, n_edges, max_deg, mean_deg


def get_moving_containers_place(
    model,
    prev_duals: Dict[Any, float],
    tol: float,
    tol_move: float,
    working_df: pd.DataFrame
) -> Tuple[List[Any], int, int, int, float]:
    """Select containers to move from placement conflict graph."""
    g = get_conflict_graph(model, prev_duals, tol)
    n_nodes, n_edges = g.number_of_nodes(), g.number_of_edges()
    degrees = sorted(g.degree(), key=lambda x: x[1], reverse=True)
    if not degrees:
        return [], n_nodes, n_edges, 0, 0.0
    max_deg = degrees[0][1]
    mean_deg = sum(d for _, d in degrees) / len(degrees)

    moving = []
    budget = len(model.dict_id_c) * tol_move
    while degrees and len(moving) < budget:
        cid, deg = degrees[0]
        if deg > 1:
            moving.append(cid)
            g.remove_node(cid)
        else:
            partner = next(iter(g.neighbors(cid)))
            to_move = get_container_tomove(model, cid, partner, working_df)
            moving.append(to_move)
            g.remove_node(cid)
            if partner in g:
                g.remove_node(partner)
        g.remove_nodes_from(nx.isolates(g))
        degrees = sorted(g.degree(), key=lambda x: x[1], reverse=True)

    return moving, n_nodes, n_edges, max_deg, mean_deg


def get_far_container(
    model,
    c1: Any,
    c2: Any,
    df_clust: pd.DataFrame,
    profiles: np.ndarray
) -> Any:
    """Pick between two containers by variance from cluster profile."""
    nf, hf, tf, metric = (
        model.indiv_field,
        model.host_field,
        model.tick_field,
        model.metrics[0]
    )
    host = df_clust.loc[df_clust[nf] == c1, hf].iloc[0]
    node_ts = (
        df_clust[df_clust[hf] == host]
        .groupby(tf)[metric].sum().sort_index().values
    )
    c1_ts = df_clust[df_clust[nf] == c1][metric].sort_index().values
    c2_ts = df_clust[df_clust[nf] == c2][metric].sort_index().values
    return c1 if np.var(node_ts - c1_ts) > np.var(node_ts - c2_ts) else c2


def get_container_tomove(
    model,
    c1: Any,
    c2: Any,
    working_df: pd.DataFrame
) -> Any:
    """Pick between two containers by variance on placement node."""
    nf, hf, tf, metric = (
        model.indiv_field,
        model.host_field,
        model.tick_field,
        model.metrics[0]
    )
    host = working_df.loc[working_df[nf] == c1, hf].iloc[0]
    node_ts = (
        working_df[working_df[hf] == host]
        .groupby(tf)[metric].sum().sort_index().values
    )
    c1_ts = working_df[working_df[nf] == c1][metric].sort_index().values
    c2_ts = working_df[working_df[nf] == c2][metric].sort_index().values
    return c1 if np.var(node_ts - c1_ts) > np.var(node_ts - c2_ts) else c2


def eval_bilevel_step(
    instance,
    labels,
    placement,
    prev_snapshot: Optional[EvalSnapshot],
    tick: int,
) -> Tuple[EvalSnapshot, Dict[str, Any]]:
    """
    Evaluate the two-stage result at this tick, comparing to previous snapshot.

    Returns:
        (snapshot, metrics_dict)
    """
    labels_now = _safe_labels_array(labels)
    placement_now = placement or {}

    # ---- Compute KPIs
    clustering_kpi = _kpi_clustering(
        labels_now=labels_now,
        labels_prev=(prev_snapshot.labels if prev_snapshot else None),
    )
    placement_kpi = _kpi_placement(
        placement_now=placement_now,
        placement_prev=(prev_snapshot.placement if prev_snapshot else None),
    )

    # Optional: incorporate similarity/adjacency metrics if useful
    # (kept here for parity with your previous evaluator helpers)
    # u = build_adjacency_matrix(labels_now)        # np.ndarray n x n
    # sim = build_similarity_matrix(instance.df_indiv, instance.metrics)
    # attr = build_matrix_indiv_attr(instance.df_indiv, instance.indiv_field)
    # You can add metrics using these matrices and merge into metrics below.

    metrics: Dict[str, Any] = {
        'tick': tick,
        **{f'clst__{k}': v for k, v in clustering_kpi.items()},
        **{f'plc__{k}': v for k, v in placement_kpi.items()},
    }

    snapshot = EvalSnapshot(
        labels=labels_now,
        placement=placement_now,
        tick=tick,
    )
    return snapshot, metrics
