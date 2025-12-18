"""Provide placement heuristics and all placement-related methods."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hots.core.interfaces import ProblemPlugin

import numpy as np

import pandas as pd


@dataclass
class _Fields:
    indiv: str
    host: str
    tick: str
    metric: str  # we use the first metric


# ---------- module helpers ----------------------------------------------------
def _get_ts(x: Any, t: Optional[int] = None) -> np.ndarray:
    """Return a 1D numpy array time series.

    - If `x` is a scalar, broadcast to length T (must be provided).
    - If `x` is already a sequence/array, convert to np.ndarray and slice to T.
    """
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(x, dtype=float)
        if t is not None:
            arr = arr[:t]
        return arr.astype(float, copy=False)
    # scalar
    if t is None:
        raise ValueError('Cannot broadcast scalar without t')
    return np.full(t, float(x), dtype=float)


def _fits(remaining: np.ndarray, demand: np.ndarray) -> bool:
    """Return True if `demand` fits into `remaining` on every tick."""
    return np.all(remaining - demand >= 0.0)


def _safe_first(values: Iterable[Any], default: Any = None) -> Any:
    """Return the first element of an iterable, or default if empty."""
    for v in values:
        return v
    return default
# -----------------------------------------------------------------------------


class PlacementPlugin(ProblemPlugin):
    """Handles the 'placement' business problem."""

    # ---------- lifecycle -----------------------------------------------------
    def __init__(self, instance):
        """Initialize the PlacementPlugin with the given instance."""
        self.instance = instance
        cfg = instance.config.connector.parameters
        self._f = _Fields(
            indiv=cfg.get('individual_field'),
            host=cfg.get('host_field'),
            tick=cfg.get('tick_field'),
            metric=cfg.get('metrics')[0],
        )
        self.pending_changes = {}

    # ---------- ProblemPlugin API --------------------------------------------
    def adjust(self, moving: List[Any], working_df):
        """Finalize move list by finding new targets and apply moves."""
        tick = self._f.tick
        tmin, tmax = int(working_df[tick].min()), int(working_df[tick].max())
        order = getattr(self, 'params', {}).get('order', 'max')

        moves = self.move_list_containers(
            self.instance, working_df, moving, tmin, tmax, order=order
        )
        return moves

    def initial(
        self,
        labels,
        df_indiv: pd.DataFrame,
        df_host: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """From initial labels, get first placement solution."""
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
        # 2) Turn it into a similarity (distance) matrix
        w = build_similarity_matrix(mat)

        return self.allocation_distant_pairwise(
            df_indiv, df_host, w, labels.tolist()
        )

    @staticmethod
    def _window_frames(inst, working_df: pd.DataFrame, tmin: int, tmax: int
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        cfg = inst.config.connector.parameters
        nf, hf, tf = cfg.get('individual_field'), cfg.get('host_field'), cfg.get('tick_field')
        metric = cfg.get('metrics')[0]

        indiv = working_df[[nf, hf, tf, metric]]
        host = inst.df_host[[hf, tf, metric]]

        mask_indiv = (indiv[tf] >= tmin) & (indiv[tf] <= tmax)
        mask_host = (host[tf] >= tmin) & (host[tf] <= tmax)

        w_indiv = indiv.loc[mask_indiv]
        w_host = host.loc[mask_host]

        ticks = np.sort(w_host[tf].unique())
        return w_indiv, w_host, ticks

    @staticmethod
    def _container_ts(w_indiv: pd.DataFrame, ticks: np.ndarray,
                      inst, container_id) -> pd.Series:
        cfg = inst.config.connector.parameters
        tf = cfg.get('tick_field')
        nf = cfg.get('individual_field')
        metric = cfg.get('metrics')[0]

        return (
            w_indiv[w_indiv[nf] == container_id]
            .groupby(tf, sort=True)[metric].sum()
            .reindex(ticks, fill_value=0.0)
        )

    @staticmethod
    def _candidate_nodes(w_indiv: pd.DataFrame, inst, old_id) -> List:
        hf = inst.config.connector.parameters.get('host_field')
        nodes = w_indiv[hf].unique()
        return [n for n in nodes if n != old_id]

    @staticmethod
    def _capacity_map(inst) -> Dict:
        hf = inst.config.connector.parameters.get('host_field')
        metric = inst.config.connector.parameters.get('metrics')[0]
        # capacity per node assumed scalar here
        return inst.df_meta.set_index(hf)[metric].to_dict()

    @staticmethod
    def _node_ts(w_host: pd.DataFrame, inst, node, ticks: np.ndarray) -> pd.Series:
        cfg = inst.config.connector.parameters
        tf = cfg.get('tick_field')
        hf = cfg.get('host_field')
        metric = cfg.get('metrics')[0]

        return (
            w_host[w_host[hf] == node]
            .groupby(tf, sort=True)[metric].sum()
            .reindex(ticks, fill_value=0.0)
        )

    def _choose_best_existing_node(
        self,
        candidates: List,
        w_host: pd.DataFrame,
        ticks: np.ndarray,
        caps: Dict,
        c_ts: pd.Series,
        inst,
    ):
        best_node, best_var = None, np.inf
        for node in candidates:
            cap_node = caps.get(node)
            if cap_node is None:
                continue

            node_ts = self._node_ts(w_host, inst, node, ticks)
            combined = node_ts + c_ts

            # feasibility check
            if np.any(combined.values > cap_node):
                continue

            v = float(combined.var(ddof=0))
            if v < best_var:
                best_var, best_node = v, node
        return best_node

    @staticmethod
    def _pick_new_node(inst, used_nodes: set, fallback):
        hf = inst.config.connector.parameters.get('host_field')
        for nid in inst.df_meta[hf].unique():
            if nid not in used_nodes:
                return nid
        return fallback

    # --- simplified public method ---------------------------------------------

    def move_container(
        self,
        inst,
        container_id,
        tmin: int,
        tmax: int,
        old_id,
        working_df: pd.DataFrame,
    ):
        """
        Move `container_id` to the best node in [tmin, tmax], choosing a feasible
        candidate (capacity per tick) that minimizes the variance of
        (node_ts + container_ts). Falls back to opening a new node if needed.
        Returns a move dict or None.
        """
        hf = inst.config.connector.parameters.get('host_field')
        logging.info(f'Moving container: {container_id}')

        w_indiv, w_host, ticks = self._window_frames(inst, working_df, tmin, tmax)
        if ticks.size == 0:
            return None  # no data in window

        c_ts = self._container_ts(w_indiv, ticks, inst, container_id)
        candidates = self._candidate_nodes(w_indiv, inst, old_id)
        caps = self._capacity_map(inst)

        best_node = self._choose_best_existing_node(
            candidates=candidates,
            w_host=w_host,
            ticks=ticks,
            caps=caps,
            c_ts=c_ts,
            inst=inst,
        )

        if best_node is None:
            logging.info(
                f'Impossible to move {container_id} on existing nodes. We need to open a new node.'
            )
            in_use = set(w_indiv[hf].unique().tolist())
            best_node = self._pick_new_node(inst, in_use, fallback=old_id)
            if best_node == old_id:
                logging.info('Impossible to open a new node: we keep the old node.')

        logging.info(f'He can go on {best_node} (old is {old_id})')
        if best_node != old_id:
            return {
                'container_name': container_id,
                'old_host': old_id,
                'new_host': best_node,
            }
        return None

    def move_list_containers(
        self, instance: Any, working_df, moving: List[int],
        tmin: int, tmax: int, order: str = 'max'
    ) -> List[Dict[str, Any]]:
        """Move the list of containers to move.

        Process:
        1. Remove all moving containers from their nodes first
        2. Store their old host IDs
        3. Order containers by consumption (max or mean)
        4. Reassign each container to best available node

        :param instance: Problem instance
        :param moving: List of container indices to move
        :param tmin: Minimum window time
        :param tmax: Maximum window time
        :param order: Order to consider ('max' or 'mean'), defaults to 'max'
        :return: List of move dicts
        """
        f = self._f
        moves_list: List[Dict[str, Any]] = []
        old_ids: Dict[int, Any] = {}
        logging.info('List of moving containers (placement):')
        logging.info(moving)
        # Step 1: Remove all moving containers and store old hosts
        for mvg_cont in moving:
            old_host = _safe_first(
                working_df.loc[
                    working_df[f.indiv] == mvg_cont
                ][f.host].to_numpy()
            )
            old_ids[mvg_cont] = old_host
            # Remove from tracking (capacity will be freed)

        # Step 2: Compute consumption for each container
        mvg_conts_cons: Dict[Any, np.ndarray] = {}
        for mvg_cont in moving:
            mvg_conts_cons[mvg_cont] = working_df.loc[
                working_df[f.indiv] == mvg_cont
            ][f.metric].to_numpy()

        # Step 3: Order containers by consumption
        if order == 'max':
            order_indivs = ((float(np.max(cons)), c) for c, cons in mvg_conts_cons.items())
        elif order == 'mean':
            order_indivs = ((float(np.mean(cons)), c) for c, cons in mvg_conts_cons.items())
        else:
            order_indivs = ((0.0, c) for c in mvg_conts_cons.keys())

        # Step 4: Move each container to best destination
        for temp, mvg_cont in sorted(order_indivs, reverse=True):
            move = self.move_container(
                instance, mvg_cont, tmin, tmax, old_ids[mvg_cont], working_df
            )
            if move is not None:
                moves_list.append(move)

        return moves_list

    # ---------- time-series helpers ------------------------------------------
    def _time_horizon(
        self, df_indiv: pd.DataFrame, df_host_meta: pd.DataFrame, time_horizon: Optional[int]
    ) -> int:
        """Infer the working time horizon length (number of ticks)."""
        if time_horizon is not None:
            return int(time_horizon)
        f = self._f
        c_first = _safe_first(df_indiv[f.metric].to_numpy())
        if isinstance(c_first, (list, tuple, np.ndarray, pd.Series)):
            return len(np.asarray(c_first))
        h_first = _safe_first(df_host_meta[f.metric].to_numpy())
        if isinstance(h_first, (list, tuple, np.ndarray, pd.Series)):
            return len(np.asarray(h_first))
        return 1

    def _capacity_for_host(self, host_row: pd.Series, t: int) -> np.ndarray:
        """Host capacity time-series (length t)."""
        return _get_ts(host_row[self._f.metric], t)

    def _demand_for_container(self, indiv_row: pd.Series, t: int) -> np.ndarray:
        """Container demand time-series (length t)."""
        return _get_ts(indiv_row[self._f.metric], t)

    # ---------- compact state builders ----------------------------------------
    def _init_remaining(self, df_host_meta: pd.DataFrame, t: int) -> List[Dict[str, Any]]:
        """List of hosts with remaining capacity timelines."""
        f = self._f
        hosts = []
        for _, h in df_host_meta.iterrows():
            hid = h[f.host]
            rem = self._capacity_for_host(h, t).copy()
            hosts.append({'id': hid, 'remaining': rem})
        return hosts

    def _current_host_map(self) -> Dict[Any, Any]:
        """Map container -> current host (from instance.df_indiv)."""
        f = self._f
        return {r[f.indiv]: r[f.host] for _, r in self.instance.df_indiv.iterrows()}

    def _compute_indiv_demands(self, df_indiv: pd.DataFrame, t: int) -> Dict[Any, np.ndarray]:
        """Map container -> demand timeline."""
        f = self._f
        return {r[f.indiv]: self._demand_for_container(r, t) for _, r in df_indiv.iterrows()}

    def _compute_host_remaining(
        self,
        df_host_meta: pd.DataFrame,
        indiv_dem: Dict[Any, np.ndarray],
        t: int,
    ) -> Dict[Any, np.ndarray]:
        """Map host -> remaining capacity timeline after subtracting current placements."""
        f = self._f
        remaining: Dict[Any, np.ndarray] = {
            h[f.host]: self._capacity_for_host(h, t) for _, h in df_host_meta.iterrows()
        }
        for _, r in self.instance.df_indiv.iterrows():
            hid = r[f.host]
            cid = r[f.indiv]
            d = indiv_dem.get(cid) or self._demand_for_container(r, t)
            remaining[hid] = remaining[hid] - d
        return remaining

    def _hosts_by_tightness(self, remaining: Dict[Any, np.ndarray]) -> List[Tuple[Any, np.ndarray]]:
        """Hosts sorted by increasing minimum remaining capacity (most constrained first)."""
        return sorted(remaining.items(), key=lambda kv: float(np.min(kv[1])))

    def _heaviest_containers_on(self, host_id: Any, t: int) -> List[Tuple[Any, float]]:
        """Containers on a host, ordered by descending peak demand over horizon."""
        f = self._f
        rows = self.instance.df_indiv[self.instance.df_indiv[f.host] == host_id]
        if rows.empty:
            return []
        peaks = rows[f.metric].map(lambda a: float(np.max(_get_ts(a, t))))
        order = rows.assign(_peak=peaks).sort_values('_peak', ascending=False)
        return [(r[f.indiv], float(p)) for (_, r), p in zip(order.iterrows(), order['_peak'])]

    def _best_destination(
            self, remaining: Dict[Any, np.ndarray], src_host: Any, demand: np.ndarray
    ) -> Optional[Any]:
        """Feasible destination host with maximum slack after placing `demand`."""
        candidate = None
        best_slack = None
        for dst, rdst in remaining.items():
            if dst == src_host:
                continue
            if not _fits(rdst, demand):
                continue
            slack = float(np.min(rdst - demand))
            if best_slack is None or slack > best_slack:
                best_slack = slack
                candidate = dst
        return candidate

    def _load_at_tick(self, series_or_array: Any, tick: int) -> float:
        """Scalar load at a given tick from a time-series-like value."""
        try:
            ts = _get_ts(series_or_array, None)
            return float(ts[min(max(tick, 0), len(ts) - 1)])
        except Exception:
            return float(series_or_array)

    def _host_remaining_over_horizon(
        self, df_host_meta: pd.DataFrame, df_indiv: pd.DataFrame
    ) -> Dict[Any, np.ndarray]:
        """Remaining capacity time-series for each host over the model horizon."""
        f = self._f
        t = self._time_horizon(df_indiv, df_host_meta, None)
        remaining: Dict[Any, np.ndarray] = {
            h[f.host]: self._capacity_for_host(h, t) for _, h in df_host_meta.iterrows()
        }
        for _, r in self.instance.df_indiv.iterrows():
            hid = r[f.host]
            demand = self._demand_for_container(r, t)
            remaining[hid] = remaining[hid] - demand
        return remaining

    def _host_remaining_at_tick(
        self, df_host_meta: pd.DataFrame, df_indiv: pd.DataFrame, tick: int
    ) -> Dict[Any, float]:
        """Remaining capacity per host at a given tick (scalar)."""
        t = self._time_horizon(df_indiv, df_host_meta, None)
        rem_ts = self._host_remaining_over_horizon(df_host_meta, df_indiv)
        idx = min(max(tick, 0), t - 1)
        return {h: float(rem[idx]) for h, rem in rem_ts.items()}

    # ---------- allocation strategies ----------------------------------------
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
        t = self._time_horizon(df_indiv, df_host_meta, time_horizon)

        hosts = self._init_remaining(df_host_meta, t)
        current_host = self._current_host_map()

        items: List[Tuple[Any, np.ndarray]] = []
        for _, r in df_indiv.iterrows():
            cid = r[f.indiv]
            d = self._demand_for_container(r, t)
            items.append((cid, d))
        items.sort(key=lambda x: (float(np.max(x[1])), float(np.sum(x[1]))), reverse=True)

        moves: List[Dict[str, Any]] = []
        for cid, demand in items:
            for h in hosts:
                if _fits(h['remaining'], demand):
                    src = current_host.get(cid, None)
                    dst = h['id']
                    if src != dst:
                        moves.append({'container': cid, 'src': src, 'dst': dst})
                    h['remaining'] = h['remaining'] - demand
                    break  # placed
        return moves

    def allocation_spread(
        self,
        df_indiv: pd.DataFrame,
        df_host_meta: pd.DataFrame,
        labels: Optional[pd.Series] = None,
    ) -> List[Dict[str, Any]]:
        """Greedy spread over the full horizon.

        Move one heaviest container off each most-constrained host to the feasible
        host that maximizes slack after placement. One move per source host.
        """
        t = self._time_horizon(df_indiv, df_host_meta, None)
        indiv_dem = self._compute_indiv_demands(df_indiv, t)
        host_remaining = self._compute_host_remaining(df_host_meta, indiv_dem, t)

        moves: List[Dict[str, Any]] = []
        for worst_host, _ in self._hosts_by_tightness(host_remaining):
            for cid, _ in self._heaviest_containers_on(worst_host, t):
                d = indiv_dem[cid]
                dst = self._best_destination(host_remaining, worst_host, d)
                if dst is None:
                    continue
                moves.append({'container': cid, 'src': worst_host, 'dst': dst})
                host_remaining[worst_host] = host_remaining[worst_host] + d
                host_remaining[dst] = host_remaining[dst] - d
                break  # one move per (worst) host per pass
        return moves

    def allocation_distant_pairwise(
        self,
        df_indiv: pd.DataFrame,
        df_host_meta: pd.DataFrame,
        distance_mat: np.ndarray,
        labels: Sequence[Any],
        lower_bound: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Place containers by repeatedly pairing the **most distant** clusters
        and assigning their items together when possible.
        """
        t = self._time_horizon(df_indiv, df_host_meta, None)
        hosts = self._init_remaining(df_host_meta, t)
        current_host = self._current_host_map()
        cid_to_demand = self._compute_indiv_demands(df_indiv, t)

        label_to_conts, label_order = self._group_by_labels_series_or_list(labels, df_indiv)
        remaining = set(range(len(label_order)))
        moves: list[Dict[str, Any]] = []

        while remaining:
            if len(remaining) == 1:
                self._spread_last_cluster_pairwise(
                    remaining, label_order, label_to_conts,
                    cid_to_demand, hosts, current_host, moves
                )
                break

            pair = self._select_best_distant_pair(remaining, distance_mat, lower_bound)

            if pair is None:
                self._spread_all_remaining_pairwise(
                    remaining, label_order, label_to_conts,
                    cid_to_demand, hosts, current_host, moves
                )
                break

            self._process_cluster_pair(
                pair, label_order, label_to_conts, cid_to_demand, hosts, current_host, moves
            )
            remaining.discard(pair[0])
            remaining.discard(pair[1])

        return moves

    def _try_place_single_pairwise(
        self, cid: Any, cid_to_demand: dict, hosts: list, current_host: dict, moves: list
    ) -> None:
        """Try to place a single container on any available host."""
        d = cid_to_demand[cid]
        for i in range(len(hosts)):
            rem = hosts[i]['remaining']
            if _fits(rem, d):
                src = current_host.get(cid, None)
                dst = hosts[i]['id']
                if src != dst:
                    moves.append({'container': cid, 'src': src, 'dst': dst})
                hosts[i]['remaining'] = rem - d
                return

    def _find_host_for_pair_pairwise(
        self, d_i: np.ndarray, d_j: np.ndarray, hosts: list
    ) -> Optional[int]:
        """Find a host that can fit both containers."""
        for i in range(len(hosts)):
            if _fits(hosts[i]['remaining'], d_i + d_j):
                return i
        return None

    def _place_pair_pairwise(
        self, c_i: Any, c_j: Any, cid_to_demand: dict, hosts: list, current_host: dict, moves: list
    ) -> None:
        """Place a pair of containers on the same host if possible."""
        di, dj = cid_to_demand[c_i], cid_to_demand[c_j]
        idx = self._find_host_for_pair_pairwise(di, dj, hosts)

        if idx is None:
            self._try_place_single_pairwise(c_i, cid_to_demand, hosts, current_host, moves)
            self._try_place_single_pairwise(c_j, cid_to_demand, hosts, current_host, moves)
            return

        dst = hosts[idx]['id']
        for cid, d in ((c_i, di), (c_j, dj)):
            src = current_host.get(cid, None)
            if src != dst:
                moves.append({'container': cid, 'src': src, 'dst': dst})
        hosts[idx]['remaining'] = hosts[idx]['remaining'] - (di + dj)

    def _select_best_distant_pair(
        self, rem_set: set, distance_mat: np.ndarray, lower_bound: Optional[float]
    ) -> Optional[Tuple[int, int]]:
        """Select the pair of clusters with the largest distance."""
        best, best_val = None, -np.inf
        rem_list = list(rem_set)

        for i in range(len(rem_list)):
            for j in range(i + 1, len(rem_list)):
                a, b = rem_list[i], rem_list[j]
                val = float(distance_mat[a, b])

                if np.isnan(val) or (lower_bound is not None and val < lower_bound):
                    continue
                if val > best_val:
                    best_val, best = val, (a, b)
        return best

    def _get_sorted_containers_pairwise(
        self, label: Any, label_to_conts: dict, cid_to_demand: dict
    ) -> list:
        """Get containers for a label, sorted by demand (descending)."""
        conts = list(label_to_conts.get(label, []))
        conts.sort(key=lambda c: float(np.max(cid_to_demand[c])), reverse=True)
        return conts

    def _spread_last_cluster_pairwise(
        self, remaining: set, label_order: list, label_to_conts: dict,
        cid_to_demand: dict, hosts: list, current_host: dict, moves: list
    ) -> None:
        """Spread the last remaining cluster across hosts."""
        last_idx = next(iter(remaining))
        lab = label_order[last_idx]
        conts = self._get_sorted_containers_pairwise(lab, label_to_conts, cid_to_demand)

        for cid in conts:
            self._try_place_single_pairwise(cid, cid_to_demand, hosts, current_host, moves)

    def _spread_all_remaining_pairwise(
        self, remaining: set, label_order: list, label_to_conts: dict,
        cid_to_demand: dict, hosts: list, current_host: dict, moves: list
    ) -> None:
        """Spread all remaining clusters when no valid pairs exist."""
        for idx in list(remaining):
            lab = label_order[idx]
            conts = self._get_sorted_containers_pairwise(lab, label_to_conts, cid_to_demand)
            for cid in conts:
                self._try_place_single_pairwise(cid, cid_to_demand, hosts, current_host, moves)

    def _process_cluster_pair(
        self, pair: Tuple[int, int], label_order: list, label_to_conts: dict,
        cid_to_demand: dict, hosts: list, current_host: dict, moves: list
    ) -> None:
        """Process a pair of clusters by placing their containers."""
        i_idx, j_idx = pair
        li, lj = label_order[i_idx], label_order[j_idx]
        list_i = list(label_to_conts.get(li, []))
        list_j = list(label_to_conts.get(lj, []))
        n_pairs = min(len(list_i), len(list_j))

        # Place pairs
        for k in range(n_pairs):
            self._place_pair_pairwise(
                list_i[k], list_j[k], cid_to_demand, hosts, current_host, moves
            )

        # Place remaining containers
        self._place_remaining_containers_pairwise(
            list_i[n_pairs:], cid_to_demand, hosts, current_host, moves
        )
        self._place_remaining_containers_pairwise(
            list_j[n_pairs:], cid_to_demand, hosts, current_host, moves
        )

    def _place_remaining_containers_pairwise(
        self, container_list: list, cid_to_demand: dict,
        hosts: list, current_host: dict, moves: list
    ) -> None:
        """Place remaining containers sorted by demand."""
        sorted_conts = sorted(
            container_list, key=lambda c: float(np.max(cid_to_demand[c])), reverse=True
        )
        for cid in sorted_conts:
            self._try_place_single_pairwise(cid, cid_to_demand, hosts, current_host, moves)

    def spread_containers(
        self, df_indiv: pd.DataFrame, tick: int, *, single_move: bool = False
    ) -> List[Dict[str, Any]]:
        """Greedy spread **at a single tick**."""
        df = self.instance.df_indiv
        df_host_meta = getattr(self.instance, 'df_host_meta', None)

        if df_host_meta is None or df.empty:
            return []

        moves: List[Dict[str, Any]] = []
        rem = self._host_remaining_at_tick(df_host_meta, df_indiv, tick)

        while True:
            moved = self._try_spread_one_container(df, rem, tick, moves)
            if not moved or single_move:
                break

        return moves

    def _try_spread_one_container(
        self, df: pd.DataFrame, rem: Dict[Any, float], tick: int, moves: list
    ) -> bool:
        """Try to spread one container from the most loaded host. Returns True if moved."""
        f = self._f
        worst_host = min(rem.items(), key=lambda kv: kv[1])[0]
        subset = df[df[f.host] == worst_host]

        if subset.empty:
            return False

        subset = self._add_load_column(subset, tick)

        for _, r in subset.iterrows():
            cid = r[f.indiv]
            load = self._load_at_tick(r[f.metric], tick)
            dst = self._find_best_destination_spread(load, worst_host, rem)

            if dst is None:
                continue

            moves.append({'container': cid, 'src': worst_host, 'dst': dst})
            rem[worst_host] += load
            rem[dst] -= load
            df.loc[df[f.indiv] == cid, f.host] = dst
            return True

        return False

    def _add_load_column(self, subset: pd.DataFrame, tick: int) -> pd.DataFrame:
        """Add load column and sort by descending load."""
        f = self._f
        subset = subset.assign(
            _load=subset[f.metric].map(lambda a: self._load_at_tick(a, tick))
        )
        return subset.sort_values('_load', ascending=False)

    def _find_best_destination_spread(
        self, load: float, src_host: Any, rem: Dict[Any, float]
    ) -> Optional[Any]:
        """Find the best destination host for a container with given load."""
        dst, best_slack = None, None
        for h, slack in rem.items():
            if h == src_host or slack < load:
                continue
            s = slack - load
            if best_slack is None or s > best_slack:
                best_slack, dst = s, h
        return dst

    # Backward-compat thin wrapper (kept to avoid breaking callers)
    def spread_containers_new(self, df_indiv: pd.DataFrame, tick: int) -> List[Dict[str, Any]]:
        """Single step of `spread_containers` (kept for compatibility)."""
        return self.spread_containers(df_indiv, tick, single_move=True)

    # ----- Clustering colocalization (merged / simplified) --------------------
    def _group_by_labels_series_or_list(
        self, labels: Sequence[Any] | pd.Series, df_indiv: pd.DataFrame
    ) -> Tuple[Dict[Any, List[Any]], List[Any]]:
        """Return (label -> [container_id]), and ordered list of unique labels.

        Works if `labels` is:
          - Series indexed by container ids,
          - Series aligned with df_indiv rows,
          - list/ndarray aligned with df_indiv rows.
        """
        f = self._f
        label_to_conts: Dict[Any, List[Any]] = {}
        if isinstance(labels, pd.Series):
            if set(labels.index) >= set(df_indiv[f.indiv]):
                for _, r in df_indiv.iterrows():
                    cid = r[f.indiv]
                    lab = labels.loc[cid]
                    label_to_conts.setdefault(lab, []).append(cid)
            else:
                for cid, lab in zip(df_indiv[f.indiv].tolist(), labels.to_numpy()):
                    label_to_conts.setdefault(lab, []).append(cid)
        else:
            labs = np.asarray(labels)
            for cid, lab in zip(df_indiv[f.indiv].tolist(), labs):
                label_to_conts.setdefault(lab, []).append(cid)
        uniq = list(label_to_conts.keys())
        return label_to_conts, uniq

    def colocalize_clusters(
        self, df_indiv: pd.DataFrame, df_host_meta: pd.DataFrame, labels: pd.Series, n_clusters: int
    ) -> List[Dict[str, Any]]:
        """Colocalize cluster members using **majority-host** heuristic.

        For each cluster:
          1) Identify the majority host (host with most members of that cluster).
          2) Move minority members to that host if they fit over the full horizon.
        Containers are processed in descending peak demand to increase feasibility.
        """
        t = self._time_horizon(df_indiv, df_host_meta, None)
        remaining = self._host_remaining_over_horizon(df_host_meta, df_indiv)
        df = self.instance.df_indiv
        cluster_of: Dict[Any, int] = labels.to_dict()

        # host -> members and majority host per cluster
        host_members: Dict[Any, List[Any]] = {
            h: df[df[self._f.host] == h][self._f.indiv].tolist() for h in remaining
        }
        cluster_counts: Dict[int, Dict[Any, int]] = {}
        for h, members in host_members.items():
            for cid in members:
                c = cluster_of.get(cid, -1)
                cluster_counts.setdefault(c, {}).setdefault(h, 0)
                cluster_counts[c][h] += 1
        majority_host: Dict[int, Any] = {
            c: max(counts.items(), key=lambda kv: kv[1])[0] for c,
            counts in cluster_counts.items() if counts
        }

        # precompute demand & peaks
        demand = {r[self._f.indiv]: self._demand_for_container(r, t) for _, r in df.iterrows()}
        peak = {cid: float(np.max(d)) for cid, d in demand.items()}

        # minority members → try to move to their cluster majority host
        candidates: List[Tuple[Any, Any]] = []
        for cid, cl in cluster_of.items():
            tgt = majority_host.get(cl)
            if tgt is None:
                continue
            cur_vals = df.loc[df[self._f.indiv] == cid, self._f.host].values
            if len(cur_vals) and cur_vals[0] != tgt:
                candidates.append((cid, tgt))
        candidates.sort(key=lambda x: peak.get(x[0], 0.0), reverse=True)

        moves: List[Dict[str, Any]] = []
        for cid, tgt in candidates:
            cur_vals = df.loc[df[self._f.indiv] == cid, self._f.host].values
            if len(cur_vals) == 0:
                continue
            src = cur_vals[0]
            d = demand[cid]
            if _fits(remaining[tgt], d):
                moves.append({'container': cid, 'src': src, 'dst': tgt})
                remaining[src] = remaining[src] + d
                remaining[tgt] = remaining[tgt] - d
                df.loc[df[self._f.indiv] == cid, self._f.host] = tgt
        return moves

    # Backward-compat thin wrapper (kept to avoid breaking callers)
    def colocalize_clusters_new(
        self, df_indiv: pd.DataFrame, df_host_meta: pd.DataFrame, labels: pd.Series, n_clusters: int
    ) -> List[Dict[str, Any]]:
        """Compatibility wrapper for the merged `colocalize_clusters`."""
        return self.colocalize_clusters(df_indiv, df_host_meta, labels, n_clusters)

    def place_opposite_clusters(
        self,
        df_indiv: pd.DataFrame,
        df_host_meta: pd.DataFrame,
        labels: pd.Series,
        clusters_to_separate: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """Enforce strong separation for a set of 'opposite' clusters."""
        t = self._time_horizon(df_indiv, df_host_meta, None)
        remaining = self._host_remaining_over_horizon(df_host_meta, df_indiv)
        df = self.instance.df_indiv
        f = self._f
        cl_of: Dict[Any, int] = labels.to_dict()
        target_set = set(clusters_to_separate)

        host_members = {h: df[df[f.host] == h][f.indiv].tolist() for h in remaining}
        anchors = self._find_cluster_anchors(host_members, cl_of, target_set)
        demand = {r[f.indiv]: self._demand_for_container(r, t) for _, r in df.iterrows()}
        peak = {cid: float(np.max(d)) for cid, d in demand.items()}
        moves: List[Dict[str, Any]] = []

        self._pull_members_to_anchors(
            anchors, cl_of, df, remaining, host_members, demand, peak, moves
        )
        self._push_foreign_from_anchors(
            anchors, target_set, cl_of, host_members, df, remaining, demand, peak, moves
        )

        return moves

    def _find_cluster_anchors(
        self, host_members: Dict[Any, List[Any]], cl_of: Dict[Any, int], target_set: set
    ) -> Dict[int, Any]:
        """Find anchor host for each target cluster (host with most members)."""
        anchors: Dict[int, Any] = {}
        for c in target_set:
            best_h, best_cnt = None, -1
            for h, members in host_members.items():
                cnt = sum(1 for cid in members if cl_of.get(cid, -1) == c)
                if cnt > best_cnt:
                    best_cnt, best_h = cnt, h
            if best_h is not None:
                anchors[c] = best_h
        return anchors

    def _pull_members_to_anchors(
        self, anchors: Dict[int, Any], cl_of: Dict[Any, int], df: pd.DataFrame,
        remaining: Dict[Any, np.ndarray], host_members: Dict[Any, List[Any]],
        demand: Dict[Any, np.ndarray], peak: Dict[Any, float], moves: list
    ) -> None:
        """Pull minority members to their cluster anchor."""
        f = self._f
        for c, anchor in anchors.items():
            members_c = [cid for cid, lab in cl_of.items() if lab == c]
            members_c.sort(key=lambda x: peak.get(x, 0.0), reverse=True)

            for cid in members_c:
                cur = _safe_first(df.loc[df[f.indiv] == cid][f.host].to_numpy())
                if cur == anchor:
                    continue

                d = demand[cid]
                if _fits(remaining[anchor], d):
                    moves.append({'container': cid, 'src': cur, 'dst': anchor})
                    remaining[cur] = remaining[cur] + d
                    remaining[anchor] = remaining[anchor] - d
                    host_members[cur].remove(cid)
                    host_members[anchor].append(cid)
                    df.loc[df[f.indiv] == cid, f.host] = anchor

    def _push_foreign_from_anchors(
        self, anchors: Dict[int, Any], target_set: set, cl_of: Dict[Any, int],
        host_members: Dict[Any, List[Any]], df: pd.DataFrame,
        remaining: Dict[Any, np.ndarray], demand: Dict[Any, np.ndarray],
        peak: Dict[Any, float], moves: list
    ) -> None:
        """Push foreign target cluster members out of anchors."""
        f = self._f
        anchor_values = set(anchors.values())

        for c, anchor in anchors.items():
            foreign = self._find_foreign_members(anchor, c, host_members, cl_of, target_set, peak)

            for cid in foreign:
                d = demand[cid]
                best_dst = self._find_best_destination_opposite(
                    anchor, anchor_values, remaining, d
                )

                if best_dst is None:
                    continue

                moves.append({'container': cid, 'src': anchor, 'dst': best_dst})
                remaining[anchor] = remaining[anchor] + d
                remaining[best_dst] = remaining[best_dst] - d
                host_members[anchor].remove(cid)
                host_members[best_dst].append(cid)
                df.loc[df[self._f.indiv] == cid, f.host] = best_dst

    def _find_foreign_members(
        self, anchor: Any, cluster: int, host_members: Dict[Any, List[Any]],
        cl_of: Dict[Any, int], target_set: set, peak: Dict[Any, float]
    ) -> list:
        """Find and sort foreign target cluster members on an anchor host."""
        foreign = [
            cid for cid in list(host_members[anchor])
            if cl_of.get(cid, -1) in target_set and cl_of.get(cid, -1) != cluster
        ]
        foreign.sort(key=lambda x: peak.get(x, 0.0), reverse=True)
        return foreign

    def _find_best_destination_opposite(
        self, anchor: Any, anchor_values: set, remaining: Dict[Any, np.ndarray], d: np.ndarray
    ) -> Optional[Any]:
        """Find best destination host avoiding anchors."""
        best_dst, best_slack = None, None
        for h, rem in remaining.items():
            if h == anchor or h in anchor_values:
                continue
            if _fits(rem, d):
                slack = float(np.min(rem - d))
                if best_slack is None or slack > best_slack:
                    best_slack, best_dst = slack, h
        return best_dst

    def find_substitution(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Find a beneficial two-way swap between hosts."""
        df_indiv = kwargs.get('df_indiv', getattr(self.instance, 'df_indiv', None))
        df_host_meta = kwargs.get('df_host_meta', getattr(self.instance, 'df_host_meta', None))

        if df_indiv is None or df_host_meta is None:
            return []

        t = self._time_horizon(df_indiv, df_host_meta, None)
        remaining = self._host_remaining_over_horizon(df_host_meta, df_indiv)
        f = self._f

        indiv_dem = {r[f.indiv]: self._demand_for_container(r, t) for _, r in df_indiv.iterrows()}
        by_host = {
            h: self.instance.df_indiv[self.instance.df_indiv[f.host] == h] for h in remaining
        }

        best_swap = self._find_best_swap(by_host, remaining, indiv_dem, f)

        if not best_swap:
            return []

        return self._create_swap_moves(best_swap)

    def _find_best_swap(
        self, by_host: dict, remaining: Dict[Any, np.ndarray],
        indiv_dem: dict, f: _Fields
    ) -> Optional[Tuple[Any, Any, Any, Any]]:
        """Find the best container swap that improves worst slack."""
        best_gain, best_swap = 0.0, None

        for ha, rows_a in by_host.items():
            if rows_a.empty:
                continue
            for hb, rows_b in by_host.items():
                if hb == ha or rows_b.empty:
                    continue

                base_slack = self._worst_slack(ha, hb, remaining)
                swap = self._find_best_pair_swap(
                    rows_a, rows_b, ha, hb, indiv_dem, remaining, base_slack, f
                )

                if swap and swap[0] > best_gain:
                    best_gain, best_swap = swap[0], swap[1:]

        return best_swap

    def _find_best_pair_swap(
        self, rows_a: pd.DataFrame, rows_b: pd.DataFrame, ha: Any, hb: Any,
        indiv_dem: dict, remaining: Dict[Any, np.ndarray], base_slack: float, f: _Fields
    ) -> Optional[Tuple[float, Any, Any, Any, Any]]:
        """Find best swap between two specific hosts."""
        best_gain, best = 0.0, None

        for _, ra in rows_a.iterrows():
            da = indiv_dem[ra[f.indiv]]
            for _, rb in rows_b.iterrows():
                db = indiv_dem[rb[f.indiv]]

                gain = self._evaluate_swap(ha, hb, da, db, remaining, base_slack)
                if gain > best_gain:
                    best_gain = gain
                    best = (best_gain, ra[f.indiv], ha, rb[f.indiv], hb)

        return best

    def _evaluate_swap(
        self, ha: Any, hb: Any, da: np.ndarray, db: np.ndarray,
        remaining: Dict[Any, np.ndarray], base_slack: float
    ) -> float:
        """Evaluate the gain from swapping two containers."""
        raft_a = remaining[ha] + da - db
        raft_b = remaining[hb] + db - da

        if np.any(raft_a < 0) or np.any(raft_b < 0):
            return 0.0

        return float(min(np.min(raft_a), np.min(raft_b))) - base_slack

    def _worst_slack(self, a: Any, b: Any, rem: Dict[Any, np.ndarray]) -> float:
        """Calculate worst slack between two hosts."""
        return float(min(np.min(rem[a]), np.min(rem[b])))

    def _create_swap_moves(self, swap: Tuple[Any, Any, Any, Any]) -> List[Dict[str, Any]]:
        """Create move list from swap tuple."""
        a_c, ha, b_c, hb = swap
        return [
            {'container': a_c, 'src': ha, 'dst': hb},
            {'container': b_c, 'src': hb, 'dst': ha},
        ]

    # ----- Graph helpers ------------------------------------------------------
    def build_place_adj_matrix(
        self, df_indiv: pd.DataFrame, id_map: Dict[int, Any]
    ) -> np.ndarray:
        """Build adjacency matrix where A[i,j] = 1 if containers i and j share host."""
        f = self._f
        hosts = {}
        for cid, i in id_map.items():
            row = df_indiv[df_indiv[f.indiv] == cid]
            hosts[i] = _safe_first(row[f.host].to_numpy())
        n = len(id_map)
        a = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if hosts.get(i) == hosts.get(j) and hosts.get(i) is not None:
                    a[i, j] = 1
                    a[j, i] = 1
        return a
