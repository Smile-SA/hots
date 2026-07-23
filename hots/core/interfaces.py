# hots/core/interfaces.py

"""HOTS core interfaces: plugin base classes.

These ABCs declare the full surface the app loop and evaluator rely on, so an
external plugin author has a contract to implement against rather than having to
reverse-engineer the pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class ClusteringPlugin(ABC):
    """Interface for clustering plugins.

    Beyond :meth:`fit`, the pipeline (``builder`` and ``evaluator``) reads and
    writes the matrix attributes below on the plugin instance. They are declared
    here so their role is explicit; subclasses should call ``super().__init__()``.
    """

    def __init__(self) -> None:
        """Initialize the shared clustering state used across the pipeline."""
        # Latest cluster labels, aligned with the instance ID map (index -> label).
        self.labels: np.ndarray | None = None
        # container x time feature matrix (with a trailing "cluster" column).
        self.clust_mat: pd.DataFrame | None = None
        # Pairwise same-cluster adjacency (u) and similarity (w) matrices.
        self.u_mat: np.ndarray | None = None
        self.w_mat: np.ndarray | None = None
        # Mean feature profile per cluster.
        self.profiles: np.ndarray | None = None

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> pd.Series:
        """Perform clustering and return labels indexed by container ID."""
        raise NotImplementedError

    def build_matrices(
        self,
        df: pd.DataFrame,
        tick_field: str,
        indiv_field: str,
        metrics: list[str],
        id_map: dict[Any, int],
        new_containers: bool = False,
    ) -> None:
        """Refresh ``clust_mat``/``labels``/``u_mat``/``w_mat`` for the window.

        Previously a free function in ``builder`` mutated the plugin from the
        outside; the plugin now owns its own state and merely reuses the pure
        matrix helpers.
        """
        # Local import keeps core decoupled from the plugins package at import time.
        from hots.plugins.clustering.builder import (
            assign_new_containers_to_nearest_cluster,
            build_adjacency_matrix,
            build_matrix_indiv_attr,
            build_similarity_matrix,
        )

        self.clust_mat = build_matrix_indiv_attr(df, tick_field, indiv_field, metrics, id_map)

        inv = {v: k for k, v in id_map.items()}
        labeled_idx = [inv[i] for i in range(len(self.labels)) if i in inv]
        labels_s = pd.Series(self.labels[: len(labeled_idx)], index=labeled_idx, name="cluster")
        self.clust_mat["cluster"] = -1
        self.clust_mat.loc[labels_s.index, "cluster"] = labels_s

        if new_containers:
            self.clust_mat = assign_new_containers_to_nearest_cluster(self.clust_mat)
            # copy=True: change_clustering writes labels back in place.
            self.labels = self.clust_mat["cluster"].astype(int).to_numpy(copy=True)

        self.u_mat = build_adjacency_matrix(self.labels)
        self.w_mat = build_similarity_matrix(self.clust_mat)


class OptimizationPlugin(ABC):
    """Interface for optimization backends (Pyomo, OR-Tools, ...).

    The two-level pipeline depends on warm-started re-solving and dual values, so
    the dual/adjacency/resize methods are part of the required surface, not just
    ``build``/``solve``. ``last_duals`` holds the most recent solve's duals.

    The evaluator also mutates the persistent model in place, one Param at a
    time (``update_w``/``update_sol_u``/``update_obj_place``), rather than
    calling ``build`` again on every no-new-container tick -- that would
    rebuild the whole abstract model from scratch for a solve that only
    changes a handful of parameter values. Trade-off: on a degenerate LP this
    can change which tied dual solution the solver returns relative to a
    from-scratch rebuild, which conflict detection is sensitive to -- see the
    reproducibility note on ``PyomoModel.fill_dual_values``.
    """

    #: Mapping constraint index -> dual value from the most recent solve.
    last_duals: dict[Any, float] | None = None

    @abstractmethod
    def build(self, *, u_mat=None, w_mat=None, v_mat=None, dv_mat=None):
        """Create and store a ready-to-solve model from the given matrices."""
        raise NotImplementedError

    @abstractmethod
    def solve(self, *, solver=None):
        """Solve the current model instance."""
        raise NotImplementedError

    @abstractmethod
    def fill_dual_values(self) -> None:
        """Extract duals for the coupling constraints into ``last_duals``."""
        raise NotImplementedError

    @abstractmethod
    def update_adjacency_constraints(self, matrix) -> None:
        """Re-fix the must-link constraints from a new adjacency matrix in place."""
        raise NotImplementedError

    @abstractmethod
    def update_size_model(
        self, dict_id_c, new_df_indiv=None, *, u_mat=None, w_mat=None, v_mat=None, dv_mat=None
    ) -> None:
        """Rebuild the model when the set of containers changes."""
        raise NotImplementedError

    @abstractmethod
    def update_w(self, w) -> None:
        """Update the clustering similarity param (pb 1) from a new w matrix in place."""
        raise NotImplementedError

    @abstractmethod
    def update_sol_u(self, u) -> None:
        """Update the clustering-adjacency param shared by both models in place.

        Distinct from ``update_adjacency_constraints``: for the placement
        model (pb 2), ``sol_u`` only feeds the objective, not the must-link
        constraint (that's ``sol_v``, via ``update_adjacency_constraints``).
        """
        raise NotImplementedError

    @abstractmethod
    def update_obj_place(self, dv) -> None:
        """Update the placement (pb 2) objective's dv param in place."""
        raise NotImplementedError


class ConnectorPlugin(ABC):
    """Interface for connector plugins (data source + move sink).

    A connector both feeds telemetry into HOTS (``load_initial``/``load_next``)
    and applies the resulting moves (``apply_moves``). ``tick_increment`` is the
    number of ticks the streaming loop advances its window by each iteration.
    """

    #: How many ticks each ``load_next`` advances the window by.
    tick_increment: int = 1

    @abstractmethod
    def load_initial(self) -> tuple[pd.DataFrame, Any, pd.DataFrame]:
        """Load the bootstrap window as ``(df_indiv, _, df_meta)``."""
        raise NotImplementedError

    @abstractmethod
    def load_next(self) -> pd.DataFrame | None:
        """Load the next batch of telemetry, or ``None`` when the source is exhausted."""
        raise NotImplementedError

    @abstractmethod
    def apply_moves(self, moves: Any, current_time: int | None = None) -> None:
        """Apply relocation moves to the target environment."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources. Default is a no-op."""
        return None


class ProblemPlugin(ABC):
    """Interface for domain‑specific problem plugins (placement, allocation, etc.)."""

    @abstractmethod
    def adjust(self, solution: Any, working_df: pd.DataFrame) -> Any:
        """Given selected moving containers and the window, return the move list."""
        raise NotImplementedError

    @abstractmethod
    def initial(
        self,
        labels: pd.Series,
        df_indiv: pd.DataFrame,
        df_host: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Produce the initial problem solution if needed.

        :param labels: cluster labels from initial clustering
        :param df_indiv: individual‐level data
        :param df_host: host‐level data
        :return: list of move dicts to apply
        """
        raise NotImplementedError

    @abstractmethod
    def build_place_adj_matrix(self, df_indiv: pd.DataFrame, id_map: dict[Any, int]) -> np.ndarray:
        """Build the same-host adjacency matrix keyed on the instance ID map."""
        raise NotImplementedError
