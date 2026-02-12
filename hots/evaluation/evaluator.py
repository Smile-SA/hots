"""Evaluation utilities for HOTS."""

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from hots.plugins.clustering.builder import (
    build_adjacency_matrix,
    build_post_clust_matrices,
    build_pre_clust_matrices,
    change_clustering,
    cluster_mean_profile,
    get_far_container,
)
from hots.utils.tools import check_missing_entries_df


@dataclass(frozen=True)
class EvalSnapshot:
    """All the info we need to compare two consecutive evaluations."""

    labels: np.ndarray  # shape (n_indiv,)
    placement: dict[Any, Any]  # mapping indiv/container -> host/node
    tick: int  # current time tick for traceability


def _safe_labels_array(labels) -> np.ndarray:
    """Ensure labels are a 1D numpy array."""
    if isinstance(labels, np.ndarray):
        return labels
    # pandas Series or list-like
    return np.asarray(labels).reshape(-1)


def _kpi_clustering(labels_now: np.ndarray, labels_prev: np.ndarray | None) -> dict[str, float]:
    """Compare clustering KPIs + delta vs previous."""
    kpi: dict[str, float] = {
        "clusters_now": float(len(np.unique(labels_now))),
    }
    if labels_prev is None:
        kpi.update(
            {
                "clusters_prev": 0.0,
                "clustered_change_ratio": 0.0,
                "reassigned_ratio": 0.0,
            }
        )
        return kpi

    labels_prev = _safe_labels_array(labels_prev)
    same_len = min(len(labels_prev), len(labels_now))
    reassigned = np.sum(labels_prev[:same_len] != labels_now[:same_len])
    kpi.update(
        {
            "clusters_prev": float(len(np.unique(labels_prev))),
            "clustered_change_ratio": float(
                abs(len(np.unique(labels_now)) - len(np.unique(labels_prev)))
            ),
            "reassigned_ratio": float(reassigned) / float(same_len) if same_len else 0.0,
        }
    )
    return kpi


def _kpi_placement(
    placement_now: dict[Any, Any],
    placement_prev: dict[Any, Any] | None,
) -> dict[str, float]:
    """Placement KPIs + delta vs previous (e.g., % moved)."""
    moved_ratio = 0.0
    if placement_prev:
        common = set(placement_now.keys()) & set(placement_prev.keys())
        moved = sum(1 for k in common if placement_now[k] != placement_prev[k])
        moved_ratio = float(moved) / float(len(common)) if common else 0.0

    return {
        "assigned_now": float(len(placement_now)),
        "assigned_prev": float(len(placement_prev) if placement_prev else 0),
        "moved_ratio": moved_ratio,
    }


def _gini(x: np.ndarray) -> float:
    """Gini coefficient for non-negative values."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    s = x.sum()
    if s == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    # Gini = (2*sum(i*x_i)/(n*sum(x))) - (n+1)/n
    i = np.arange(1, n + 1, dtype=float)
    return float((2.0 * (i * x).sum()) / (n * s) - (n + 1.0) / n)


def _entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy (natural log)."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def _kpi_cluster_structure(labels_now: np.ndarray) -> dict[str, float]:
    """Cluster size distribution KPIs (balance, inequality)."""
    labels_now = _safe_labels_array(labels_now)
    if labels_now.size == 0:
        return {
            "n_clusters": 0.0,
            "singleton_ratio": 0.0,
            "size_min": 0.0,
            "size_max": 0.0,
            "size_mean": 0.0,
            "size_std": 0.0,
            "size_entropy": 0.0,
            "size_gini": 0.0,
        }

    c = Counter(labels_now.tolist())
    sizes = np.asarray(list(c.values()), dtype=float)
    n_clusters = float(len(sizes))
    singleton_ratio = float(np.mean(sizes == 1.0)) if sizes.size else 0.0

    return {
        "n_clusters": n_clusters,
        "singleton_ratio": singleton_ratio,
        "size_min": float(sizes.min()) if sizes.size else 0.0,
        "size_max": float(sizes.max()) if sizes.size else 0.0,
        "size_mean": float(sizes.mean()) if sizes.size else 0.0,
        "size_std": float(sizes.std(ddof=0)) if sizes.size else 0.0,
        "size_entropy": _entropy_from_counts(sizes),
        "size_gini": _gini(sizes),
    }


def _kpi_conflict_graph(g: nx.Graph) -> dict[str, float]:
    """Graph structure + weight summary KPIs."""
    n = g.number_of_nodes()
    m = g.number_of_edges()
    if n == 0:
        return {
            "nodes": 0.0,
            "edges": 0.0,
            "density": 0.0,
            "components": 0.0,
            "largest_component_ratio": 0.0,
            "deg_max": 0.0,
            "deg_mean": 0.0,
            "deg_p95": 0.0,
            "deg_std": 0.0,
            "w_sum": 0.0,
            "w_mean": 0.0,
            "w_p95": 0.0,
            "w_max": 0.0,
        }

    degrees = np.asarray([d for _, d in g.degree()], dtype=float)
    deg_max = float(degrees.max()) if degrees.size else 0.0
    deg_mean = float(degrees.mean()) if degrees.size else 0.0
    deg_p95 = float(np.percentile(degrees, 95)) if degrees.size else 0.0
    deg_std = float(degrees.std(ddof=0)) if degrees.size else 0.0

    # Edge weights (some edges might not have weight if added elsewhere)
    w = []
    for _, _, data in g.edges(data=True):
        if "weight" in data:
            w.append(float(data["weight"]))
    w = np.asarray(w, dtype=float)

    # Components
    comp_sizes = [len(c) for c in nx.connected_components(g)] if n else []
    largest_ratio = (max(comp_sizes) / n) if comp_sizes and n else 0.0

    return {
        "nodes": float(n),
        "edges": float(m),
        "density": float(nx.density(g)) if n > 1 else 0.0,
        "components": float(len(comp_sizes)),
        "largest_component_ratio": float(largest_ratio),
        "deg_max": deg_max,
        "deg_mean": deg_mean,
        "deg_p95": deg_p95,
        "deg_std": deg_std,
        "w_sum": float(w.sum()) if w.size else 0.0,
        "w_mean": float(w.mean()) if w.size else 0.0,
        "w_p95": float(np.percentile(w, 95)) if w.size else 0.0,
        "w_max": float(w.max()) if w.size else 0.0,
    }


def _kpi_migrations_from_moves(
    moves: Any,
    *,
    n_indiv: int,
) -> dict[str, float]:
    """KPIs from the return of problem.adjust()."""
    # In your flow, `moves` is usually a list of moved container ids.
    if moves is None:
        k = 0
    elif isinstance(moves, (list, tuple, set)):
        k = len(moves)
    else:
        # fallback: unknown structure
        try:
            k = len(moves)  # type: ignore[arg-type]
        except Exception:
            k = 0

    return {
        "moves_count": float(k),
        "moves_ratio": float(k) / float(n_indiv) if n_indiv else 0.0,
    }


def _kpi_load_balance(
    working_df: pd.DataFrame,
    *,
    nf: str,
    hf: str,
    metrics: list[str],
) -> dict[str, float]:
    """
    Generic load-balance KPIs over hosts for a chosen metric.

    NOTE: This operates on `working_df` as provided (whatever your current window is).
    """
    if working_df is None or working_df.empty or not metrics:
        return {
            "host_load_mean": 0.0,
            "host_load_std": 0.0,
            "host_load_cv": 0.0,
            "host_load_p95": 0.0,
            "host_load_max": 0.0,
            "host_load_gini": 0.0,
            "n_hosts": 0.0,
            "n_indiv": 0.0,
        }

    metric = metrics[0]
    if metric not in working_df.columns or hf not in working_df.columns:
        return {
            "host_load_mean": 0.0,
            "host_load_std": 0.0,
            "host_load_cv": 0.0,
            "host_load_p95": 0.0,
            "host_load_max": 0.0,
            "host_load_gini": 0.0,
            "n_hosts": float(working_df[hf].nunique()) if hf in working_df.columns else 0.0,
            "n_indiv": float(working_df[nf].nunique()) if nf in working_df.columns else 0.0,
        }

    host_load = working_df.groupby(hf, sort=False)[metric].sum().astype(float).values
    if host_load.size == 0:
        host_load = np.asarray([0.0])

    mean = float(host_load.mean())
    std = float(host_load.std(ddof=0))
    cv = float(std / mean) if mean != 0.0 else 0.0

    return {
        "host_load_mean": mean,
        "host_load_std": std,
        "host_load_cv": cv,
        "host_load_p95": float(np.percentile(host_load, 95)),
        "host_load_max": float(host_load.max()),
        "host_load_gini": _gini(host_load),
        "n_hosts": float(working_df[hf].nunique()),
        "n_indiv": float(working_df[nf].nunique()) if nf in working_df.columns else 0.0,
    }


def _placement_from_working_df(
    working_df: pd.DataFrame,
    *,
    nf: str,
    hf: str,
    tf: str,
    tick: int | None = None,
) -> dict[Any, Any]:
    """
    Derive container -> host mapping from working_df.

    Strategy:
      - use rows at `tick` if provided, otherwise use max tick in df
      - if multiple rows per (container, tick), take the most frequent host
    """
    if working_df is None or working_df.empty:
        return {}

    if nf not in working_df.columns or hf not in working_df.columns or tf not in working_df.columns:
        return {}

    t = tick if tick is not None else int(working_df[tf].max())
    df_t = working_df.loc[working_df[tf] == t, [nf, hf]].copy()
    if df_t.empty:
        return {}

    # choose most frequent host per container (robust to duplicates)
    def _mode_host(s: pd.Series) -> Any:
        vc = s.value_counts(dropna=False)
        return vc.index[0] if not vc.empty else None

    plc = df_t.groupby(nf, sort=False)[hf].apply(_mode_host).dropna()
    return plc.to_dict()


def eval_solutions(
    instance,
    clustering,
    clust_opt,
    problem_opt,
    problem,
    working_df,
    *,
    prev_snapshot: EvalSnapshot | None = None,
    tick: int | None = None,
) -> dict[str, Any]:
    """Run the evaluation pipeline and collect evaluation metrics."""
    # 1) Build & solve clustering problem
    cfg = instance.config.connector.parameters
    nf, hf, tf = cfg.get("individual_field"), cfg.get("host_field"), cfg.get("tick_field")
    metrics = cfg.get("metrics")

    new_containers = False
    if len(clustering.labels) < working_df[nf].nunique():
        working_df = check_missing_entries_df(working_df, tf, nf, hf, metrics)
        new_containers = True

    build_pre_clust_matrices(
        working_df, tf, nf, metrics, instance.get_id_map(), clustering, new_containers
    )

    if new_containers:
        logging.info("\n🔍 New containers detected: updating optimization models 🔍\n")
        clust_opt.update_size_model(
            instance.get_id_map(), working_df, u_mat=clustering.u_mat, w_mat=clustering.w_mat
        )
    else:
        clust_opt.build(u_mat=clustering.u_mat, w_mat=clustering.w_mat)
    clust_opt.solve(
        solver=instance.config.optimization.parameters.get("solver", "glpk"),
    )

    # 3) Extract dual values
    prev_duals = clust_opt.last_duals
    clust_opt.fill_dual_values()

    # 4) Read tolerances
    tol = instance.config.problem.parameters.get("tol", 0.1)
    tol_move = instance.config.problem.parameters.get("tol_move", 0.1)

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
    g_clust = get_conflict_graph(clust_opt.last_duals, prev_duals, tol)
    clust_conf_kpi = _kpi_conflict_graph(g_clust)

    (clustering.clust_mat, clust_nb_changes) = change_clustering(
        moving, clustering, instance.get_id_map()
    )
    clustering.u_mat = build_adjacency_matrix(clustering.labels)
    clust_opt.update_adjacency_constraints(clustering.u_mat)
    clust_opt.solve(
        solver=instance.config.optimization.parameters.get("solver", "glpk"),
    )
    clust_opt.fill_dual_values()

    # 6) Build & solve business problem
    v_mat = problem.build_place_adj_matrix(working_df, instance.get_id_map())
    dv_mat = build_post_clust_matrices(clustering.clust_mat)

    if new_containers:
        problem_opt.update_size_model(
            instance.get_id_map(), working_df, u_mat=clustering.u_mat, v_mat=v_mat, dv_mat=dv_mat
        )
    else:
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
    g_place = get_conflict_graph(problem_opt.last_duals, prev_duals, tol)
    place_conf_kpi = _kpi_conflict_graph(g_place)

    # 8) Apply business‑problem changes
    moves = problem.adjust(moving, working_df)
    moves_kpi = _kpi_migrations_from_moves(moves, n_indiv=len(instance.get_id_map()))
    v_mat = problem.build_place_adj_matrix(working_df, instance.get_id_map())
    problem_opt.update_adjacency_constraints(v_mat)
    problem_opt.solve()
    problem_opt.fill_dual_values()

    # 9) Collect metrics
    metrics: dict[str, Any] = {
        # clustering conflict
        **{f"clust_conf__{k}": v for k, v in clust_conf_kpi.items()},
        # placement conflict
        **{f"place_conf__{k}": v for k, v in place_conf_kpi.items()},
        # moves applied by adjust()
        **{f"moves__{k}": v for k, v in moves_kpi.items()},
        "moving_containers": moves,
    }

    if (
        tick is None
        and working_df is not None
        and not working_df.empty
        and tf in working_df.columns
    ):
        tick = int(working_df[tf].max())
    tick = int(tick) if tick is not None else -1

    placement_now = _placement_from_working_df(
        working_df,
        nf=nf,
        hf=hf,
        tf=tf,
        tick=tick,
    )

    snapshot, step_kpi = eval_step_kpis(
        instance,
        labels=clustering.labels,
        placement=placement_now,
        prev_snapshot=prev_snapshot,
        tick=tick,
        working_df=working_df,
    )
    metrics.update(step_kpi)

    return moves, metrics, snapshot


def get_conflict_graph(
    cur_duals: dict[Any, float], prev_duals: dict[Any, float], tol: float
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
    cur_duals: dict[Any, float],
    prev_duals: dict[Any, float],
    tol: float,
    tol_move: float,
    df_clust: pd.DataFrame,
    profiles: np.ndarray,
) -> tuple[list[Any], int, int, int, float]:
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
    instance,
    cur_duals,
    prev_duals: dict[Any, float],
    tol: float,
    tol_move: float,
    working_df: pd.DataFrame,
) -> tuple[list[Any], int, int, int, float]:
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
    nf, hf, tf = cfg.get("individual_field"), cfg.get("host_field"), cfg.get("tick_field")
    metric = cfg.get("metrics")[0]

    df = working_df[[nf, hf, tf, metric]].copy()

    # Determine c1's host (fallback: pick c2 if c1 has no rows)
    host_rows = df.loc[df[nf] == c1, hf]
    if host_rows.empty:
        return c2
    host = host_rows.iloc[0]

    # Node time series on c1's host, indexed by time
    node_ts = df[df[hf] == host].groupby(tf, sort=True)[metric].sum().sort_index()

    idx = node_ts.index

    def cont_ts(cid):
        s = df[df[nf] == cid].groupby(tf, sort=True)[metric].sum().sort_index()
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


def eval_step_kpis(
    instance,
    labels,
    placement,
    prev_snapshot: EvalSnapshot | None,
    tick: int,
    *,
    working_df: pd.DataFrame | None = None,
) -> tuple[EvalSnapshot, dict[str, Any]]:
    """
    Evaluate the two-stage result at this tick, comparing to previous snapshot.

    Returns:
        (snapshot, metrics_dict)
    """
    labels_now = _safe_labels_array(labels)
    placement_now = placement or {}

    # ---- existing KPIs
    clustering_kpi = _kpi_clustering(
        labels_now=labels_now,
        labels_prev=(prev_snapshot.labels if prev_snapshot else None),
    )
    placement_kpi = _kpi_placement(
        placement_now=placement_now,
        placement_prev=(prev_snapshot.placement if prev_snapshot else None),
    )

    # ---- new KPIs: cluster structure
    clst_struct_kpi = _kpi_cluster_structure(labels_now)

    # ---- new KPIs: placement delta counts
    moved_count = 0
    stable_count = 0
    if prev_snapshot and prev_snapshot.placement:
        common = set(placement_now.keys()) & set(prev_snapshot.placement.keys())
        moved_count = sum(1 for k in common if placement_now[k] != prev_snapshot.placement[k])
        stable_count = len(common) - moved_count

    placement_delta_kpi = {
        "moved_count": float(moved_count),
        "stable_count": float(stable_count),
    }

    # ---- new KPIs: load balance (optional if working_df provided)
    cfg = instance.config.connector.parameters
    nf, hf = cfg.get("individual_field"), cfg.get("host_field")
    metrics_list = cfg.get("metrics") or []
    load_kpi = {}
    if working_df is not None:
        load_kpi = _kpi_load_balance(
            working_df,
            nf=nf,
            hf=hf,
            metrics=metrics_list,
        )

    metrics: dict[str, Any] = {
        "tick": tick,
        **{f"clst__{k}": v for k, v in clustering_kpi.items()},
        **{f"clst_struct__{k}": v for k, v in clst_struct_kpi.items()},
        **{f"plc__{k}": v for k, v in placement_kpi.items()},
        **{f"plc_delta__{k}": v for k, v in placement_delta_kpi.items()},
        **{f"load__{k}": v for k, v in load_kpi.items()},
    }

    snapshot = EvalSnapshot(
        labels=labels_now,
        placement=placement_now,
        tick=tick,
    )
    return snapshot, metrics
