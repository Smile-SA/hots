"""Plotting utilities for HOTS.

Design:
- All plotting functions live here.
- A single `VisualizationConfig` controls saving, colors, layout parameters, etc.
- Each plot returns `(fig, saved_path)` where `saved_path` is a `Path` or `None`.

Saving rules:
    - If `out_path` is provided -> save there (even if `viz.save.plots_folder` is None)
    - Else save to `viz.save.plots_folder` if:
        - viz.enabled is True
        - viz.save.enabled is True
        - viz.save.plots_folder is not None
    - If saving occurs, figure is closed if `viz.save.close_on_save` is True (default).

"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hac
from matplotlib import gridspec
from matplotlib import patches as mpatches

from hots.config.loader import VisualizationConfig
from hots.core.instance import FieldNames

# Helpers


def _metric(viz: VisualizationConfig, metric: str | None) -> str:
    return metric or viz.default_metric


def _slugify(s: str) -> str:
    s = s.strip().lower()
    out: list[str] = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", "."):
            out.append("_")
    res = "".join(out)
    while "__" in res:
        res = res.replace("__", "_")
    return res.strip("_") or "plot"


def _build_filename(viz: VisualizationConfig, stem: str) -> str:
    final_stem = _slugify(stem) if viz.naming.slugify else (stem.strip() or "plot")
    final_stem = f"{viz.naming.prefix}{final_stem}{viz.naming.suffix}"
    ext = viz.save.default_format.strip(".")
    return f"{final_stem}.{ext}"


def _maybe_save(
    fig: plt.Figure,
    *,
    viz: VisualizationConfig,
    out_path: str | Path | None,
    filename: str | None,
    default_stem: str,
) -> Path | None:
    if not viz.enabled or not viz.save.enabled:
        return None

    if out_path is not None:
        path = Path(out_path)
    else:
        if viz.save.plots_folder is None:
            return None
        name = _build_filename(viz, filename or default_stem)
        path = Path(viz.save.plots_folder) / name

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=viz.save.dpi, bbox_inches=viz.save.bbox_inches)

    if viz.save.close_on_save:
        plt.close(fig)

    return path


def _cluster_colors(viz: VisualizationConfig) -> Sequence[str]:
    # supports Sequence[str] as in your config dataclass
    return list(viz.colors.clusters)


def _highlight_colors(viz: VisualizationConfig) -> Sequence[str]:
    return list(viz.colors.highlight)


# Clustering plots


def plot_clustering(
    df_clust: pd.DataFrame,
    *,
    viz: VisualizationConfig,
    dict_id_c: Mapping[int, int] | None = None,
    metric: str | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
    filename: str | None = None,
) -> tuple[plt.Figure, Path | None]:
    if not viz.enabled:
        return None, None
    metric_name = _metric(viz, metric)
    colors = _cluster_colors(viz)

    fig = plt.figure()
    fig.suptitle(title or f"{metric_name} consumption of containers grouped by cluster")

    if "cluster" not in df_clust.columns:
        raise ValueError("df_clust must contain a 'cluster' column")

    n_clusters = int(df_clust["cluster"].max()) + 1
    gs = gridspec.GridSpec(n_clusters, 1)

    values = df_clust.drop(columns=["cluster"])
    max_cons = float(values.to_numpy().max()) if not values.empty else 0.0

    inv: dict[int, int] | None = None
    if dict_id_c is not None:
        inv = {v: k for k, v in dict_id_c.items()}

    for k, data in df_clust.groupby("cluster"):
        k_int = int(k)
        ax = fig.add_subplot(gs[k_int, 0])
        ax.set_title(f"Cluster #{k_int}", pad=k_int)
        ax.set(xlabel="time", ylabel=metric_name)
        ax.grid(True)
        ax.set_ylim([0, math.ceil(max_cons) if max_cons > 0 else 1])

        for idx, row in data.drop(columns=["cluster"]).iterrows():
            label = str(inv.get(idx, idx)) if inv is not None else str(idx)
            ax.plot(row.values, color=colors[k_int % len(colors)], label=label)

        # readable legend: show only cluster size by default
        patch = mpatches.Patch(color=colors[k_int % len(colors)], label=f"{len(data)} containers")
        ax.legend(handles=[patch], loc="upper left")

    saved = _maybe_save(
        fig,
        viz=viz,
        out_path=out_path,
        filename=filename,
        default_stem=f"clustering_{metric_name}",
    )
    return fig, saved


def plot_clustering_spec_cont(
    df_clust: pd.DataFrame,
    *,
    viz: VisualizationConfig,
    containers_to_show: Sequence[str | int],
    dict_id_c: Mapping[int, int] | None = None,
    metric: str | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
    filename: str | None = None,
) -> tuple[plt.Figure, Path | None]:
    metric_name = _metric(viz, metric)
    colors = _cluster_colors(viz)
    highlight = _highlight_colors(viz)

    fig = plt.figure()
    fig.suptitle(title or f"{metric_name} consumption of containers grouped by cluster")

    if "cluster" not in df_clust.columns:
        raise ValueError("df_clust must contain a 'cluster' column")

    n_clusters = int(df_clust["cluster"].max()) + 1
    gs = gridspec.GridSpec(n_clusters, 1)

    values = df_clust.drop(columns=["cluster"])
    max_cons = float(values.to_numpy().max()) if not values.empty else 0.0

    containers_set = {str(c) for c in containers_to_show}
    inv: dict[int, int] | None = None
    if dict_id_c is not None:
        inv = {v: k for k, v in dict_id_c.items()}

    for k, data in df_clust.groupby("cluster"):
        k_int = int(k)
        ax = fig.add_subplot(gs[k_int, 0])
        ax.set_title(f"Cluster #{k_int}", pad=k_int)
        ax.set(xlabel="time", ylabel=metric_name)
        ax.grid(True)
        ax.set_ylim([0, math.ceil(max_cons) if max_cons > 0 else 1])

        for idx, row in data.drop(columns=["cluster"]).iterrows():
            label = str(inv.get(idx, idx)) if inv is not None else str(idx)
            if label in containers_set:
                ax.plot(row.values, color=highlight[k_int % len(highlight)], label=label)
            else:
                ax.plot(row.values, color=colors[k_int % len(colors)], alpha=0.6)

        ax.legend()

    saved = _maybe_save(
        fig,
        viz=viz,
        out_path=out_path,
        filename=filename,
        default_stem=f"clustering_spec_{metric_name}",
    )
    return fig, saved


def plot_containers_clustering_together(
    df_clust: pd.DataFrame,
    *,
    viz: VisualizationConfig,
    metric: str | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
    filename: str | None = None,
) -> tuple[plt.Figure, Path | None]:
    metric_name = _metric(viz, metric)
    colors = _cluster_colors(viz)

    fig, ax = plt.subplots()
    fig.suptitle(title or f"Containers clustering ({metric_name})")

    if "cluster" not in df_clust.columns:
        raise ValueError("df_clust must contain a 'cluster' column")

    for _, row in df_clust.iterrows():
        cluster = int(row["cluster"])
        values = row.drop(labels="cluster").to_numpy()
        ax.plot(values, color=colors[cluster % len(colors)], alpha=0.9)

    saved = _maybe_save(
        fig,
        viz=viz,
        out_path=out_path,
        filename=filename,
        default_stem=f"clustering_together_{metric_name}",
    )
    return fig, saved


def plot_dendrogram(
    z_all: np.ndarray,
    *,
    viz: VisualizationConfig,
    title: str | None = None,
    out_path: str | Path | None = None,
    filename: str | None = None,
) -> tuple[plt.Figure, Path | None]:
    fig = plt.figure()
    plt.title(title or "Hierarchical Clustering Dendrogram")
    plt.xlabel("sample index")
    plt.ylabel("distance")

    hac.dendrogram(
        z_all,
        leaf_rotation=viz.dendrogram.leaf_rotation,
        leaf_font_size=viz.dendrogram.leaf_font_size,
    )

    saved = _maybe_save(
        fig,
        viz=viz,
        out_path=out_path,
        filename=filename,
        default_stem="dendrogram",
    )
    return fig, saved


def plot_cluster_profiles(
    profiles: np.ndarray,
    *,
    viz: VisualizationConfig,
    title: str | None = None,
    out_path: str | Path | None = None,
    filename: str | None = None,
) -> tuple[plt.Figure, Path | None]:
    colors = _cluster_colors(viz)

    fig, ax = plt.subplots()
    fig.suptitle(title or "Cluster profiles (mean of containers)")

    for i in range(int(profiles.shape[0])):
        ax.plot(profiles[i, :], color=colors[i % len(colors)], label=str(i))
    ax.legend()

    saved = _maybe_save(
        fig,
        viz=viz,
        out_path=out_path,
        filename=filename,
        default_stem="cluster_profiles",
    )
    return fig, saved


# Node-level plots


def plot_clustering_containers_by_node(
    df_indiv: pd.DataFrame,
    *,
    viz: VisualizationConfig,
    labels: Sequence[int],
    dict_id_c: Mapping[int, int] | None,
    fields: FieldNames,
    metric: str | None = None,
    filter_biggest_cluster: bool = False,
    title: str | None = None,
    out_path: str | Path | None = None,
    filename: str | None = None,
) -> tuple[plt.Figure, Path | None]:
    metric_name = _metric(viz, metric)
    colors = _cluster_colors(viz)

    if filter_biggest_cluster and len(labels) > 0:
        to_filter = int(np.bincount(np.asarray(labels)).argmax())
    else:
        to_filter = None

    fig = plt.figure()
    fig.suptitle(title or f"{metric_name} consumption in each node (containers colored by cluster)")

    unique_hosts = int(df_indiv[fields.host].nunique())
    gs = gridspec.GridSpec(math.ceil(unique_hosts / 2), 2)

    x_ticks = np.sort(df_indiv[fields.tick].unique())

    inv: dict[int, int] | None = None
    if dict_id_c is not None:
        inv = {v: k for k, v in dict_id_c.items()}

    i = 0
    for host, data_n in df_indiv.groupby(fields.host):
        agg = pd.Series(data=[0.0] * len(x_ticks), index=x_ticks)

        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_title(str(host))
        ax.set(xlabel="time", ylabel=metric_name)
        ax.grid(True)

        for c_val, data_c in data_n.groupby(fields.individual):
            s = data_c.groupby(fields.tick)[metric_name].sum().reindex(x_ticks).fillna(0.0)
            agg = agg.add(s)

            c_int = int(c_val)
            if inv is not None:
                c_int = int(inv.get(int(c_val), c_val))

            cluster = int(labels[c_int]) if c_int < len(labels) else 0

            if to_filter is not None and cluster == to_filter:
                ax.plot(x_ticks, agg.values, color=colors[cluster % len(colors)], alpha=0.0)
            else:
                ax.plot(x_ticks, agg.values, color=colors[cluster % len(colors)], label=str(c_int))

        i += 1

    saved = _maybe_save(
        fig,
        viz=viz,
        out_path=out_path,
        filename=filename,
        default_stem=f"by_node_{metric_name}",
    )
    return fig, saved


def plot_containers_groupby_nodes(
    df_indiv: pd.DataFrame,
    *,
    viz: VisualizationConfig,
    max_cap: float,
    sep_time: float | None,
    fields: FieldNames,
    metric: str | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
    filename: str | None = None,
) -> tuple[plt.Figure | None, Path | None]:
    if not viz.enabled:
        return None, None

    metric_name = _metric(viz, metric)

    fig, ax = plt.subplots()
    fig.suptitle(title or f"Host {metric_name.upper()} consumption")

    # capacity margin controlled by config
    margin = max_cap * float(viz.nodes.capacity_margin_ratio)
    ax.set_ylim([0, max_cap + margin])

    pvt = pd.pivot_table(
        df_indiv,
        columns=fields.host,
        index=df_indiv[fields.tick],
        aggfunc="sum",
        values=metric_name,
    )
    pvt.plot(ax=ax, legend=False)

    if sep_time is not None:
        ax.axvline(x=sep_time, color=viz.nodes.sep_time_color, linestyle=viz.nodes.sep_time_style)
    ax.axhline(y=max_cap, color=viz.nodes.capacity_color)

    saved = _maybe_save(
        fig,
        viz=viz,
        out_path=out_path,
        filename=filename,
        default_stem=f"hosts_{metric_name}",
    )
    return fig, saved


# Live / incremental plots (still useful under Agg for saving snapshots)


def init_containers_plot(
    df_indiv: pd.DataFrame,
    *,
    viz: VisualizationConfig,
    sep_time: float | None,
    fields: FieldNames,
    metric: str | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    metric_name = _metric(viz, metric)

    fig, ax = plt.subplots()
    fig.suptitle(title or "Containers consumption evolution")

    ax.set_ylim([0, float(df_indiv[metric_name].max()) if not df_indiv.empty else 1.0])
    ax.set_xlim(
        [
            float(df_indiv[fields.tick].min()) if not df_indiv.empty else 0.0,
            float(df_indiv[fields.tick].max()) if not df_indiv.empty else 1.0,
        ]
    )

    df = df_indiv.loc[df_indiv[fields.tick] <= sep_time]
    pvt = pd.pivot_table(
        df,
        columns=df[fields.individual],
        index=df[fields.tick],
        aggfunc="sum",
        values=metric_name,
    )
    ax.plot(pvt)
    if sep_time is not None:
        ax.axvline(x=sep_time, color=viz.nodes.sep_time_color, linestyle=viz.nodes.sep_time_style)

    return fig, ax


def update_containers_plot(
    ax: plt.Axes,
    df_indiv: pd.DataFrame,
    *,
    viz: VisualizationConfig,
    fields: FieldNames,
    metric: str | None = None,
) -> None:
    metric_name = _metric(viz, metric)

    pvt = pd.pivot_table(
        df_indiv,
        columns=df_indiv[fields.individual],
        index=df_indiv[fields.tick],
        aggfunc="sum",
        values=metric_name,
    )
    ax.plot(pvt)
    plt.pause(viz.live.pause_seconds)


def init_nodes_plot(
    df_indiv: pd.DataFrame,
    *,
    viz: VisualizationConfig,
    dict_id_n: Mapping[int, str] | None,
    sep_time: float | None,
    max_cap: float,
    fields: FieldNames,
    metric: str | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    metric_name = _metric(viz, metric)
    colors = _cluster_colors(viz)

    fig, ax = plt.subplots()
    fig.suptitle(title or "Nodes consumption evolution")

    margin = max_cap * float(viz.nodes.capacity_margin_ratio)
    ax.set_ylim([0, max_cap + margin])

    df = df_indiv.loc[df_indiv[fields.tick] <= sep_time].copy()
    df.reset_index(drop=True, inplace=True)

    inv: dict[str, int] | None = None
    if dict_id_n is not None:
        inv = {v: int(k) for k, v in dict_id_n.items()}

    for host, data_h in df.groupby(fields.host):
        host_int = inv.get(host, 0) if inv is not None else 0
        ax.plot(
            data_h.groupby(fields.tick)[metric_name].sum(),
            color=colors[host_int % len(colors)],
        )

    if sep_time is not None:
        ax.axvline(x=sep_time, color=viz.nodes.sep_time_color, linestyle=viz.nodes.sep_time_style)
    ax.axhline(y=max_cap, color=viz.nodes.capacity_color)

    return fig, ax


def update_nodes_plot(
    ax: plt.Axes,
    df_indiv: pd.DataFrame,
    *,
    viz: VisualizationConfig,
    dict_id_n: Mapping[int, str] | None,
    fields: FieldNames,
    metric: str | None = None,
) -> None:
    metric_name = _metric(viz, metric)
    colors = _cluster_colors(viz)

    df = df_indiv.reset_index(drop=True)

    inv: dict[str, int] | None = None
    if dict_id_n is not None:
        inv = {v: int(k) for k, v in dict_id_n.items()}

    for host, data_h in df.groupby(fields.host):
        host_int = inv.get(host, 0) if inv is not None else 0
        ax.plot(
            data_h.groupby(fields.tick)[metric_name].sum(),
            color=colors[host_int % len(colors)],
        )

    plt.pause(viz.live.pause_seconds)


# Optimization / graph


def plot_conflict_graph(
    graph: nx.Graph,
    *,
    viz: VisualizationConfig,
    title: str | None = None,
    out_path: str | Path | None = None,
    filename: str | None = None,
) -> tuple[plt.Figure, Path | None]:
    fig, ax = plt.subplots()
    fig.suptitle(title or "Conflict graph")

    pos = nx.spring_layout(
        graph,
        k=viz.graphs.spring_layout.k,
        iterations=viz.graphs.spring_layout.iterations,
    )
    nx.draw(graph, pos, with_labels=True, ax=ax)

    if viz.graphs.show_edge_weights:
        weights = nx.get_edge_attributes(graph, "weight")
        if weights:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights, ax=ax)

    saved = _maybe_save(
        fig,
        viz=viz,
        out_path=out_path,
        filename=filename,
        default_stem="conflict_graph",
    )
    return fig, saved
