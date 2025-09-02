"""Provide placement heuristics and all placement-related methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from hots.core.interfaces import ProblemPlugin


@dataclass
class _Fields:
    indiv: str
    host: str
    metric: str  # we use the first metric


def _get_ts(x: Any, T: Optional[int] = None) -> np.ndarray:
    """Return a 1D numpy array time series.

    - If `x` is a scalar, broadcast to length T (must be provided).
    - If `x` is already a sequence/array, convert to np.ndarray and slice to T.
    """
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(x, dtype=float)
        if T is not None:
            arr = arr[:T]
        return arr.astype(float, copy=False)
    # scalar
    if T is None:
        raise ValueError("Cannot broadcast scalar without T")
    return np.full(T, float(x), dtype=float)


def _fits(remaining: np.ndarray, demand: np.ndarray) -> bool:
    return np.all(remaining - demand >= 0.0)


def _safe_first(values: Iterable[Any], default: Any = None) -> Any:
    for v in values:
        return v
    return default


class PlacementPlugin(ProblemPlugin):
    """Handles the 'placement' business problem."""

    def __init__(self, instance):
        self.instance = instance
        cfg = instance.config
        self._f = _Fields(
            indiv=cfg.individual_field,
            host=cfg.host_field,
            metric=cfg.metrics[0],
        )

    # ProblemPlugin required API
    def adjust(self, model: Any, moving: List[Any]) -> List[Dict[str, Any]]:
        """
        Given the Pyomo model and list of container IDs to move,
        return the final list of move‐dicts.
        """
        # start with the solver’s own moves
        moves: List[Dict[str, Any]] = model.extract_moves()

        # compute the time window
        df = self.instance.df_indiv
        tick = self.instance.config.tick_field
        tmin, tmax = int(df[tick].min()), int(df[tick].max())
        order = self.params.get('order', 'max')

        extra = self.move_list_containers(
            self.instance,
            moving,
            tmin,
            tmax,
            order=order
        )
        moves.extend(extra)
        return moves

    def initial(
        self,
        labels: pd.Series,
        df_indiv: pd.DataFrame,
        df_host: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """From initial labels, get first placement solution."""
        print(df_indiv)
        print(df_host)
        from hots.plugins.clustering.builder import (
            build_matrix_indiv_attr,
            build_similarity_matrix,
        )

        # 1) Build the (containers × features) matrix
        mat = build_matrix_indiv_attr(
            df_indiv,
            self.instance.config.tick_field,
            self.instance.config.individual_field,
            self.instance.config.metrics,
            self.instance.get_id_map(),
        )
        # 2) Turn it into a similarity matrix
        w = build_similarity_matrix(mat)

        moves = self.allocation_distant_pairwise(
            df_indiv, df_host, w, labels.values.tolist()
        )

        return moves

    # Core primitives
    def assign_container_node(
        self, node_id: Any, container_id: Any, remove: bool = True
    ) -> Dict[str, Any]:
        """Return the move dict to place `container_id` on `node_id`.

        If `remove` is True, we capture the current host as `src`; otherwise
        `src` is set to None.
        """
        f = self._f
        df_indiv = self.instance.df_indiv
        current = _safe_first(
            df_indiv.loc[df_indiv[f.indiv] == container_id][f.host].to_numpy()
        )
        if current == node_id:
            return {"container": container_id, "src": current, "dst": current}
        return {
            "container": container_id,
            "src": current if remove else None,
            "dst": node_id,
        }

    def remove_container_node(self, node_id: Any, container_id: Any) -> Dict[str, Any]:
        """Represent removing a container from a node (dst=None)."""
        return {"container": container_id, "src": node_id, "dst": None}

    def move_container(self, host_src: Any, host_dst: Any, container_id: Any) -> Dict[str, Any]:
        return {"container": container_id, "src": host_src, "dst": host_dst}

    def move_list_containers(
        self, host_src: Any, host_dst: Any, list_containers: Sequence[Any]
    ) -> List[Dict[str, Any]]:
        return [self.move_container(host_src, host_dst, cid) for cid in list_containers]

    # Helpers operating on time-series feasibility
    def _time_horizon(self, df_indiv: pd.DataFrame, df_host_meta: pd.DataFrame, time_horizon: Optional[int]) -> int:
        if time_horizon is not None:
            return int(time_horizon)
        # infer max length from first rows
        f = self._f
        c_first = _safe_first(df_indiv[f.metric].to_numpy())
        if isinstance(c_first, (list, tuple, np.ndarray, pd.Series)):
            return len(np.asarray(c_first))
        h_first = _safe_first(df_host_meta[f.metric].to_numpy())
        if isinstance(h_first, (list, tuple, np.ndarray, pd.Series)):
            return len(np.asarray(h_first))
        # fallback
        return 1

    def _capacity_for_host(self, host_row: pd.Series, T: int) -> np.ndarray:
        cap = host_row[self._f.metric]
        return _get_ts(cap, T)

    def _demand_for_container(self, indiv_row: pd.Series, T: int) -> np.ndarray:
        dem = indiv_row[self._f.metric]
        return _get_ts(dem, T)

    # Allocation strategies
    def _group_by_labels(
        self, labels: pd.Series, df_indiv: pd.DataFrame
    ) -> Tuple[Dict[Any, List[Any]], List[Any]]:
        """Return (label -> [container_id]), and ordered list of unique labels.

        Labels can be:
          - Series indexed by container ids
          - Series aligned with df_indiv rows
          - List/ndarray aligned with df_indiv rows
        """
        f = self._f
        label_to_conts: Dict[Any, List[Any]] = {}
        if isinstance(labels, pd.Series):
            if set(labels.index) >= set(df_indiv[f.indiv]):
                # index are container ids
                for _, r in df_indiv.iterrows():
                    cid = r[f.indiv]
                    lab = labels.loc[cid]
                    label_to_conts.setdefault(lab, []).append(cid)
            else:
                # align by order
                for cid, lab in zip(df_indiv[f.indiv].tolist(), labels.to_numpy()):
                    label_to_conts.setdefault(lab, []).append(cid)
        else:
            # list/ndarray
            labs = np.asarray(labels)
            for cid, lab in zip(df_indiv[f.indiv].tolist(), labs):
                label_to_conts.setdefault(lab, []).append(cid)

        uniq = list(label_to_conts.keys())
        return label_to_conts, uniq

    def allocation_distant_pairwise(
        self,
        df_indiv: pd.DataFrame,
        df_host_meta: pd.DataFrame,
        labels: pd.Series,
        distance_mat: np.ndarray,
        lower_bound: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Place containers by repeatedly picking the two **most distant** clusters
        (based on `distance_mat`) and assigning their items in pairs onto hosts.

        If a given pair cannot fit on any single host, we place them individually.
        Any remaining items from an unmatched cluster are then *spread* individually.
        """
        f = self._f
        T = self._time_horizon(df_indiv, df_host_meta, None)
        moves: List[Dict[str, Any]] = []

        # Host state
        hosts = self._init_remaining(df_host_meta, T)
        host_id_list = [h["id"] for h in hosts]
        H = len(hosts)
        if H == 0:
            return moves
        host_idx_cycle = 0  # emulate "first available node, then next"

        # Current assignment (for src in move dicts)
        current_host = self._current_host_map()

        # Prepare per-container demand lookup
        cid_to_demand: Dict[Any, np.ndarray] = {}
        for _, r in df_indiv.iterrows():
            cid_to_demand[r[f.indiv]] = self._demand_for_container(r, T)

        # Group containers by cluster label
        label_to_conts, labels_list = self._group_by_labels(labels, df_indiv)

        # Map label -> position in distance matrix
        # We assume `distance_mat` is indexed by the order of `labels_list`
        L = len(labels_list)
        if distance_mat.shape[0] != distance_mat.shape[1] or distance_mat.shape[0] != L:
            # Try to reindex by sorted unique labels if mismatched
            labels_list = sorted(labels_list)
            L = len(labels_list)
            if distance_mat.shape[0] != L:
                raise ValueError(
                    "distance_mat shape does not match number of unique labels"
                )

        # A mask of remaining labels (by index in labels_list)
        remaining = set(range(L))

        def _idx_of(label):
            return labels_list.index(label)

        def _try_place_on_host_pair(d_i: np.ndarray, d_j: np.ndarray) -> Optional[int]:
            """Return host index if both demands fit together on some host, using cyclic scan."""
            nonlocal host_idx_cycle
            for k in range(H):
                idx = (host_idx_cycle + k) % H
                rem = hosts[idx]["remaining"]
                if _fits(rem, d_i + d_j):
                    return idx
            return None

        def _place_single(cid: Any) -> Optional[int]:
            """Place a single container on first feasible host (cyclic)."""
            nonlocal host_idx_cycle
            d = cid_to_demand[cid]
            for k in range(H):
                idx = (host_idx_cycle + k) % H
                rem = hosts[idx]["remaining"]
                if _fits(rem, d):
                    # commit
                    src = current_host.get(cid, None)
                    dst = hosts[idx]["id"]
                    if src != dst:
                        moves.append({"container": cid, "src": src, "dst": dst})
                    hosts[idx]["remaining"] = rem - d
                    host_idx_cycle = (idx + 1) % H
                    return idx
            return None  # not placed

        def _place_pair(c_i: Any, c_j: Any) -> None:
            """Try to put both on same host; else place individually."""
            nonlocal host_idx_cycle
            d_i = cid_to_demand[c_i]
            d_j = cid_to_demand[c_j]
            idx = _try_place_on_host_pair(d_i, d_j)
            if idx is not None:
                # commit both to host idx
                src_i = current_host.get(c_i, None)
                src_j = current_host.get(c_j, None)
                dst = hosts[idx]["id"]
                if src_i != dst:
                    moves.append({"container": c_i, "src": src_i, "dst": dst})
                if src_j != dst:
                    moves.append({"container": c_j, "src": src_j, "dst": dst})
                hosts[idx]["remaining"] = hosts[idx]["remaining"] - (d_i + d_j)
                host_idx_cycle = (idx + 1) % H
            else:
                # place individually (best-effort)
                _place_single(c_i)
                _place_single(c_j)

        # Process until no labels remain
        while remaining:
            if len(remaining) == 1:
                # Only one cluster left: spread its containers individually
                last_idx = next(iter(remaining))
                last_label = labels_list[last_idx]
                leftover = label_to_conts.get(last_label, [])
                # Heaviest first can help
                leftover.sort(key=lambda cid: float(np.max(cid_to_demand[cid])), reverse=True)
                for cid in leftover:
                    _place_single(cid)
                remaining.clear()
                break

            # Choose the two MOST distant clusters among remaining
            # distance_mat is assumed to be symmetric, with NaN/diag ignored
            best_pair = None
            best_dist = -np.inf
            rem_list = sorted(list(remaining))
            for a in rem_list:
                for b in rem_list:
                    if a >= b:
                        continue
                    val = distance_mat[a, b]
                    if np.isnan(val):
                        continue
                    if lower_bound is not None and val < lower_bound:
                        continue
                    if val > best_dist:
                        best_dist = val
                        best_pair = (a, b)

            if best_pair is None:
                # No valid pair under lower_bound; just spread all remaining
                for idx in rem_list:
                    lab = labels_list[idx]
                    conts = label_to_conts.get(lab, [])
                    conts.sort(key=lambda cid: float(np.max(cid_to_demand[cid])), reverse=True)
                    for cid in conts:
                        _place_single(cid)
                remaining.clear()
                break

            i_idx, j_idx = best_pair
            label_i, label_j = labels_list[i_idx], labels_list[j_idx]
            list_i = list(label_to_conts.get(label_i, []))
            list_j = list(label_to_conts.get(label_j, []))

            # Pairwise place
            n_pairs = min(len(list_i), len(list_j))
            for k in range(n_pairs):
                _place_pair(list_i[k], list_j[k])

            # Place any leftovers from the larger cluster
            if len(list_i) > n_pairs:
                rem_i = list_i[n_pairs:]
                rem_i.sort(key=lambda cid: float(np.max(cid_to_demand[cid])), reverse=True)
                for cid in rem_i:
                    _place_single(cid)
            if len(list_j) > n_pairs:
                rem_j = list_j[n_pairs:]
                rem_j.sort(key=lambda cid: float(np.max(cid_to_demand[cid])), reverse=True)
                for cid in rem_j:
                    _place_single(cid)

            # Mark these two clusters as processed
            remaining.discard(i_idx)
            remaining.discard(j_idx)

        return moves

    def allocation_ffd(
        self,
        df_indiv: pd.DataFrame,
        df_host_meta: pd.DataFrame,
        time_horizon: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """First-Fit Decreasing across time-series capacity.

        Sort containers by descending peak demand; assign to first host whose
        remaining timeline supports the entire demand.
        """
        f = self._f
        T = self._time_horizon(df_indiv, df_host_meta, time_horizon)

        # hosts state: remaining capacity per host over time
        hosts = []
        for _, h in df_host_meta.iterrows():
            hid = h[f.host]
            rem = self._capacity_for_host(h, T).copy()
            hosts.append({"id": hid, "remaining": rem})
        host_ids = [h["id"] for h in hosts]

        # index current host per container
        current_host = {
            r[f.indiv]: r[f.host] for _, r in self.instance.df_indiv.iterrows()
        }

        # sort containers by descending total demand (or max)
        items: List[Tuple[Any, np.ndarray]] = []
        for _, r in df_indiv.iterrows():
            cid = r[f.indiv]
            d = self._demand_for_container(r, T)
            items.append((cid, d))
        items.sort(key=lambda x: (float(np.max(x[1])), float(np.sum(x[1]))), reverse=True)

        moves: List[Dict[str, Any]] = []
        for cid, demand in items:
            placed = False
            for h in hosts:
                if _fits(h["remaining"], demand):
                    src = current_host.get(cid, None)
                    dst = h["id"]
                    if src != dst:
                        moves.append({"container": cid, "src": src, "dst": dst})
                    h["remaining"] = h["remaining"] - demand
                    placed = True
                    break
            if not placed:
                # not placeable with current hosts – leave as is (no move)
                pass

        return moves

    def allocation_spread(
        self,
        df_indiv: pd.DataFrame,
        df_host_meta: pd.DataFrame,
        labels: Optional[pd.Series] = None,
    ) -> List[Dict[str, Any]]:
        """Greedy spread: try to reduce peak usage by moving heavy containers
        off the most loaded hosts to the least loaded feasible ones."""
        f = self._f
        # Reuse FFD notion of remaining by reconstructing from scratch
        T = self._time_horizon(df_indiv, df_host_meta, None)

        # compute total demand per container
        indiv_dem = {
            r[f.indiv]: self._demand_for_container(r, T) for _, r in df_indiv.iterrows()
        }

        # compute host remaining given current placement
        # start from full capacity
        host_remaining: Dict[Any, np.ndarray] = {}
        for _, h in df_host_meta.iterrows():
            host_remaining[h[f.host]] = self._capacity_for_host(h, T)

        # subtract each container currently assigned
        for _, r in self.instance.df_indiv.iterrows():
            hid = r[f.host]
            cid = r[f.indiv]
            d = indiv_dem.get(cid)
            if d is None:
                d = self._demand_for_container(r, T)
            host_remaining[hid] = host_remaining[hid] - d

        # order hosts by current peak usage (lowest remaining)
        hosts_sorted = sorted(
            host_remaining.items(), key=lambda kv: float(np.min(kv[1]))
        )

        moves: List[Dict[str, Any]] = []
        # iterate from most constrained to least constrained
        for worst_host, rem in hosts_sorted:
            # pick the heaviest container on worst_host
            rows = self.instance.df_indiv[self.instance.df_indiv[f.host] == worst_host]
            if rows.empty:
                continue
            # heaviest by max demand
            rows = rows.copy()
            rows["_peak"] = rows[f.metric].map(lambda a: float(np.max(_get_ts(a, T))))
            rows.sort_values("_peak", ascending=False, inplace=True)

            for _, r in rows.iterrows():
                cid = r[f.indiv]
                d = indiv_dem[cid]
                # try to move to a host with highest remaining slack where it fits
                candidate = None
                best_slack = None
                for dst, rdst in host_remaining.items():
                    if dst == worst_host:
                        continue
                    if _fits(rdst, d):
                        slack = float(np.min(rdst - d))
                        if best_slack is None or slack > best_slack:
                            best_slack = slack
                            candidate = dst
                if candidate is not None:
                    moves.append({"container": cid, "src": worst_host, "dst": candidate})
                    # update remaining
                    host_remaining[worst_host] = host_remaining[worst_host] + d
                    host_remaining[candidate] = host_remaining[candidate] - d
                    break  # take one move per host in this pass

        return moves

    # Additional helpers translated from legacy surface (lightweight stubs)
    def free_full_nodes(self) -> List[Dict[str, Any]]:
        """Try to free nodes that are over capacity by using `allocation_spread`."""
        return self.allocation_spread(self.instance.df_indiv, self.instance.df_host)

    def assign_indiv_available_host(self, container_id: Any, tick: int) -> Optional[Dict[str, Any]]:
        """Find any feasible host for a container at a given tick (one-slot view)."""
        f = self._f
        df_indiv = self.instance.df_indiv
        df_host = self.instance.df_host

        row = df_indiv[df_indiv[f.indiv] == container_id]
        if row.empty:
            return None
        demand = _get_ts(_safe_first(row[f.metric].to_numpy()), tick + 1)[tick:tick+1]

        for _, h in df_host.iterrows():
            cap = _get_ts(h[f.metric], tick + 1)[tick:tick+1]
            if _fits(cap, demand):
                cur = _safe_first(row[f.host].to_numpy())
                if cur != h[f.host]:
                    return {"container": container_id, "src": cur, "dst": h[f.host]}
                return None
        return None

    def nb_min_nodes(
        self, df_indiv: pd.DataFrame, df_host_meta: pd.DataFrame, time_horizon: Optional[int] = None
    ) -> int:
        """Lower bound on required nodes: sum demand per tick / max capacity per tick, rounded up."""
        f = self._f
        T = self._time_horizon(df_indiv, df_host_meta, time_horizon)

        total = np.zeros(T)
        for _, r in df_indiv.iterrows():
            total += self._demand_for_container(r, T)

        cap = np.zeros(T)
        for _, h in df_host_meta.iterrows():
            cap += self._capacity_for_host(h, T)

        # avoid division by zero
        cap[cap == 0] = np.inf
        need = np.max(np.ceil(total / cap))
        if not np.isfinite(need):
            return 0
        return int(need)

    # Placeholders for compatibility with prior surface. They can be
    # specialized later as needed.
    def find_substitution(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return []

    def spread_containers_new(self, df_indiv: pd.DataFrame, tick: int) -> List[Dict[str, Any]]:
        return []

    def spread_containers(self, df_indiv: pd.DataFrame, tick: int) -> List[Dict[str, Any]]:
        return []

    def colocalize_clusters(
        self, df_indiv: pd.DataFrame, df_host_meta: pd.DataFrame, labels: pd.Series, n_clusters: int
    ) -> List[Dict[str, Any]]:
        return []

    def colocalize_clusters_new(
        self, df_indiv: pd.DataFrame, df_host_meta: pd.DataFrame, labels: pd.Series, n_clusters: int
    ) -> List[Dict[str, Any]]:
        return []

    def allocation_distant_pairwise(
        self, df_indiv: pd.DataFrame, df_host_meta: pd.DataFrame, labels: pd.Series, distance_mat: np.ndarray
    ) -> List[Dict[str, Any]]:
        return []

    def place_opposite_clusters(
        self, df_indiv: pd.DataFrame, df_host_meta: pd.DataFrame, labels: pd.Series, clusters_to_separate: Sequence[int]
    ) -> List[Dict[str, Any]]:
        return []

    def build_placement_adj_matrix(self, df_indiv: pd.DataFrame, dict_id_c: Dict[int, Any]) -> np.ndarray:
        """Build adjacency matrix where A[i,j] = 1 if containers i and j share host."""
        f = self._f
        # map container index -> host
        hosts = {}
        for i, cid in dict_id_c.items():
            row = df_indiv[df_indiv[f.indiv] == cid]
            hosts[i] = _safe_first(row[f.host].to_numpy())
        n = len(dict_id_c)
        A = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if hosts.get(i) == hosts.get(j) and hosts.get(i) is not None:
                    A[i, j] = 1
                    A[j, i] = 1
        return A