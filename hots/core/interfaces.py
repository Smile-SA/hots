# hots/core/interfaces.py

"""HOTS core interfaces: plugin base classes."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import pandas as pd


class IngestionPlugin(ABC):
    """Interface for ingestion plugins."""

    @abstractmethod
    def load_initial(self) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """Load initial batch: returns individual, host, and metadata."""
        pass

    @abstractmethod
    def load_next(self) -> pd.DataFrame:
        """Load the next batch of individual-level data."""
        pass


class ClusteringPlugin(ABC):
    """Interface for clustering plugins."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> pd.Series:
        """Perform clustering and return labels."""
        pass


class OptimizationPlugin(ABC):
    """Interface for optimization plugins."""

    @abstractmethod
    def solve(self, df_host: pd.DataFrame, labels: pd.Series) -> Any:
        """Solve the optimization problem with given data and labels."""
        pass


class ConnectorPlugin(ABC):
    """Interface for connector plugins."""

    @abstractmethod
    def apply_moves(self, moves: Any) -> None:
        """Apply relocation moves to the target environment."""
        pass


class ProblemPlugin(ABC):
    """Interface for domain‑specific problem plugins (placement, allocation, etc.)."""

    @abstractmethod
    def adjust(self, solution: Any, **kwargs) -> Any:
        """
        Given an optimization solution (and any auxiliary data),
        return an adjusted solution.
        """
        pass

    def initial(
        self,
        labels: pd.Series,
        df_indiv: pd.DataFrame,
        df_host: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Produce the initial problem solution if needed.

        :param labels: cluster labels from initial clustering
        :param df_indiv: individual‐level data
        :param df_host: host‐level data
        :return: list of move dicts to apply
        """
