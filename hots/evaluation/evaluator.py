"""Evaluation utilities for HOTS."""

from typing import Any, Dict, List, Tuple

import networkx as nx

import numpy as np

import pandas as pd

from plugins.clustering.builder import build_matrix_indiv_attr

from sklearn.metrics import silhouette_score


def eval_solutions(
    df_indiv: pd.DataFrame,
    df_host: pd.DataFrame,
    labels: pd.Series,
    clustering,
    optimization,
    heuristic,
    instance
) -> Tuple[Any, Dict[str, Any]]:
    """Run the full pipeline and collect evaluation metrics."""
    # 1) Silhouette on individual‑level profiles
    mat = build_matrix_indiv_attr(
        df_indiv,
        instance.config.tick_field,
        instance.config.individual_field,
        instance.config.metrics,
        instance.get_id_map()
    )
    sil = silhouette_score(mat.values, labels)

    # 2) Solve the optimization model
    model = optimization.solve(df_host, labels)
    model.solve(
        solver=instance.config.optimization.parameters.get('solver', 'glpk'),
        verbose=False
    )

    # 3) Extract dual values
    duals = model.fill_dual_values()

    # 4) Read tolerances
    tol = instance.config.heuristic.parameters.get('tol', 0.1)
    tol_move = instance.config.heuristic.parameters.get('tol_move', 0.1)

    # 5) Conflict detection & pick moving containers
    pb = instance.config.optimization.parameters.get('pb_number', 1)
    if pb == 1:
        moving, nodes, edges, max_deg, mean_deg = (
            get_moving_containers_clust(
                model, duals, tol, tol_move,
                df_clust=df_indiv,
                profiles=None
            )
        )
    else:
        moving, nodes, edges, max_deg, mean_deg = (
            get_moving_containers_place(
                model, duals, tol, tol_move,
                working_df=df_indiv
            )
        )

    # 6) Apply heuristic
    solution2 = heuristic.adjust(model, moving)

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
