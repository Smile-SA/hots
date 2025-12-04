"""Evaluation utilities for HOTS."""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from hots.plugins.clustering.builder import (
    build_post_clust_matrices,
    build_pre_clust_matrices,
    change_clustering,
    cluster_mean_profile,
    get_far_container
)
from hots.utils.tools import check_missing_entries_df

import networkx as nx

import numpy as np

import pandas as pd


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
    instance,
    clustering,
    clust_opt,
    problem_opt,
    problem,
    working_df
) -> Dict[str, Any]:
    """Run the evaluation pipeline and collect evaluation metrics."""
    # 1) Build & solve clustering problem
    cfg = instance.config.connector.parameters
    nf, hf, tf = cfg.get('individual_field'), cfg.get('host_field'), cfg.get('tick_field')
    metrics = cfg.get('metrics')

    new_containers = False
    if len(clustering.labels) < working_df[nf].nunique():
        working_df = check_missing_entries_df(
            working_df, tf, nf, hf, metrics
        )
        new_containers = True

    build_pre_clust_matrices(
        working_df,
        tf, nf, metrics,
        instance.get_id_map(),
        clustering,
        new_containers
    )

    if new_containers:
        logging.info('\n🔍 New containers detected: updating optimization models 🔍\n')
        clust_opt.update_size_model(
            instance.get_id_map(), working_df,
            u_mat=clustering.u_mat, w_mat=clustering.w_mat
        )
    else:
        # TODO update and no build
        clust_opt.build(u_mat=clustering.u_mat, w_mat=clustering.w_mat)
    clust_opt.solve(
        solver=instance.config.optimization.parameters.get('solver', 'glpk'),
    )

    # 3) Extract dual values
    prev_duals = clust_opt.last_duals
    clust_opt.fill_dual_values()

    # 4) Read tolerances
    tol = instance.config.problem.parameters.get('tol', 0.1)
    tol_move = instance.config.problem.parameters.get('tol_move', 0.1)

    # 5) Conflict detection & pick moving containers for clustering
    clustering.profiles = cluster_mean_profile(clustering.clust_mat)
    moving, nodes, edges, max_deg, mean_deg = get_moving_containers_clust(
        clust_opt.last_duals,
        prev_duals,
        tol,
        tol_move,
        df_clust=clustering.clust_mat,
        profiles=clustering.profiles,
    )

    (clustering.clust_mat, clust_nb_changes) = change_clustering(
        moving, clustering, instance.get_id_map()
    )

    # 6) Build & solve business problem
    v_mat = problem.build_place_adj_matrix(
        working_df,
        instance.get_id_map())
    dv_mat = build_post_clust_matrices(clustering.clust_mat)

    if new_containers:
        problem_opt.update_size_model(
            instance.get_id_map(), working_df,
            u_mat=clustering.u_mat, v_mat=v_mat, dv_mat=dv_mat
        )
    else:
        # TODO update not build
        problem_opt.build(u_mat=clustering.u_mat, v_mat=v_mat, dv_mat=dv_mat)
    problem_opt.solve()
    prev_duals = problem_opt.last_duals
    problem_opt.fill_dual_values()
    # 7) Conflict detection & pick moving containers for problem
    # TODO use problem factory for this method
    moving, nodes, edges, max_deg, mean_deg = get_moving_containers_place(
        instance,
        problem_opt.last_duals,
        prev_duals,
        tol,
        tol_move,
        working_df=working_df,
    )

    # 8) Apply business‑problem changes
    moves = problem.adjust(moving, working_df)

    # 9) Update and solve opti models
    # TODO

    # 7) Collect metrics
    # TODO better handle this (clust vs place)
    metrics: Dict[str, Any] = {
        'conflict_nodes': nodes,
        'conflict_edges': edges,
        'max_conf_degree': max_deg,
        'mean_conf_degree': mean_deg,
        'moving_containers': moves
    }

    return moves, metrics


def get_conflict_graph(
    cur_duals: Dict[Any, float],
    prev_duals: Dict[Any, float],
    tol: float
) -> nx.Graph:
    """Build conflict graph where edges represent dual increases above tolerance."""
    g = nx.Graph()

    cur_duals = cur_duals or {}
    prev_duals = prev_duals or {}

    # Work with the union of keys from previous and current iterations
    all_idx = set(cur_duals.keys()) | set(prev_duals.keys())

    for idx in all_idx:
        cur = cur_duals.get(idx, 0.0)
        prev = prev_duals.get(idx, 0.0)
        delta = abs(cur - prev)

        if delta <= tol:
            continue

        # idx is usually a tuple like (c_i, c_j) for must_link constraints
        if isinstance(idx, tuple) and len(idx) == 2:
            a, b = idx
            g.add_edge(a, b, weight=delta)
        else:
            # Fallback: single node constraint, or unexpected index type
            g.add_node(idx, weight=delta)

    return g


def get_moving_containers_clust(
    cur_duals: Dict[Any, float],
    prev_duals: Dict[Any, float],
    tol: float,
    tol_move: float,
    df_clust: pd.DataFrame,
    profiles: np.ndarray
) -> Tuple[List[Any], int, int, int, float]:
    """Select containers to move from clustering conflict graph."""
    g = get_conflict_graph(cur_duals, prev_duals, tol)
    n_nodes, n_edges = g.number_of_nodes(), g.number_of_edges()
    degrees = sorted(g.degree(), key=lambda x: x[1], reverse=True)
    if not degrees:
        return [], n_nodes, n_edges, 0, 0.0
    max_deg = degrees[0][1]
    mean_deg = sum(d for _, d in degrees) / len(degrees)

    moving = []
    budget = max(0, int(math.ceil(len(df_clust) * tol_move)))
    while degrees and len(moving) < budget:
        cid, deg = degrees[0]

        if deg == 0:
            # isolated node: just drop it from the graph
            g.remove_node(cid)

        elif deg > 1:
            # high-degree node: move it
            moving.append(cid)
            g.remove_node(cid)

        else:
            # deg == 1: pick its single neighbor as partner
            partner = next(iter(g.neighbors(cid)))
            to_move = get_far_container(cid, partner, df_clust, profiles)
            moving.append(to_move)
            g.remove_node(cid)
            if partner in g:
                g.remove_node(partner)

        isolates = list(nx.isolates(g))
        if isolates:
            g.remove_nodes_from(isolates)

        # recompute degrees after mutations
        degrees = sorted(g.degree(), key=lambda x: x[1], reverse=True)

    return moving, n_nodes, n_edges, max_deg, mean_deg


def get_moving_containers_place(
    instance, cur_duals,
    prev_duals: Dict[Any, float],
    tol: float,
    tol_move: float,
    working_df: pd.DataFrame
) -> Tuple[List[Any], int, int, int, float]:
    """Select containers to move from placement conflict graph."""
    g = get_conflict_graph(cur_duals, prev_duals, tol)
    n_nodes, n_edges = g.number_of_nodes(), g.number_of_edges()
    degrees = sorted(g.degree(), key=lambda x: x[1], reverse=True)
    if not degrees:
        return [], n_nodes, n_edges, 0, 0.0
    max_deg = degrees[0][1]
    mean_deg = sum(d for _, d in degrees) / len(degrees)

    moving = []
    budget = len(instance.get_id_map()) * tol_move
    while degrees and len(moving) < budget:
        cid, deg = degrees[0]

        if deg == 0:
            # isolated node: just drop it from the graph
            g.remove_node(cid)

        elif deg > 1:
            # high-degree node: move it
            moving.append(cid)
            g.remove_node(cid)

        else:
            # deg == 1: pick its single neighbor as partner
            partner = next(iter(g.neighbors(cid)))
            to_move = get_container_tomove(instance, cid, partner, working_df)
            moving.append(to_move)
            g.remove_node(cid)
            if partner in g:
                g.remove_node(partner)

        isolates = list(nx.isolates(g))
        if isolates:
            g.remove_nodes_from(isolates)

        # recompute degrees after mutations
        degrees = sorted(g.degree(), key=lambda x: x[1], reverse=True)

    return moving, n_nodes, n_edges, max_deg, mean_deg


def get_container_tomove(instance, c1, c2, working_df: pd.DataFrame):
    """Pick between two containers by variance on c1's placement node.

    Chooses the container whose removal makes the node's time series smoother
    (i.e., smaller variance of node_ts - container_ts).
    """
    cfg = instance.config.connector.parameters
    nf, hf, tf = cfg.get('individual_field'), cfg.get('host_field'), cfg.get('tick_field')
    metric = cfg.get('metrics')[0]

    df = working_df[[nf, hf, tf, metric]].copy()

    # Determine c1's host (fallback: pick c2 if c1 has no rows)
    host_rows = df.loc[df[nf] == c1, hf]
    if host_rows.empty:
        return c2
    host = host_rows.iloc[0]

    # Node time series on c1's host, indexed by time
    node_ts = (
        df[df[hf] == host]
        .groupby(tf, sort=True)[metric]
        .sum()
        .sort_index()
    )

    idx = node_ts.index

    def cont_ts(cid):
        s = (
            df[df[nf] == cid]
            .groupby(tf, sort=True)[metric]
            .sum()
            .sort_index()
        )
        # align to node timeline; missing ticks -> 0
        return s.reindex(idx, fill_value=0)

    c1_ts = cont_ts(c1)
    c2_ts = cont_ts(c2)

    # Residual node load after removing each container
    res1 = node_ts - c1_ts
    res2 = node_ts - c2_ts

    # Population variance (ddof=0) to avoid small-sample weirdness
    var1 = float(res1.var(ddof=0))
    var2 = float(res2.var(ddof=0))

    # Pick the removal that yields *lower* variance (smoother)
    return c1 if var1 < var2 else c2


# def get_container_tomove(
#     instance,
#     c1: Any,
#     c2: Any,
#     working_df: pd.DataFrame
# ) -> Any:
#     """Pick between two containers by variance on placement node."""
#     nf, hf, tf, metric = (
#         instance.indiv_field,
#         instance.host_field,
#         instance.tick_field,
#         instance.metrics[0]
#     )
#     host = working_df.loc[working_df[nf] == c1, hf].iloc[0]
#     node_ts = (
#         working_df[working_df[hf] == host]
#         .groupby(tf)[metric].sum().sort_index().values
#     )
#     c1_ts = working_df[working_df[nf] == c1][metric].sort_index().values
#     c2_ts = working_df[working_df[nf] == c2][metric].sort_index().values
#     return c1 if np.var(node_ts - c1_ts) > np.var(node_ts - c2_ts) else c2


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
