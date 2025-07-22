# hots/core/interfaces.py

"""HOTS core interfaces: plugin base classes."""

from abc import ABC, abstractmethod
from typing import Any, Tuple

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


class HeuristicPlugin(ABC):
    """Interface for heuristic plugins."""

    @abstractmethod
    def adjust(self, solution: Any) -> Any:
        """Adjust the solution according to heuristic logic."""
        pass


class ConnectorPlugin(ABC):
    """Interface for connector plugins."""

    @abstractmethod
    def apply_moves(self, moves: Any) -> None:
        """Apply relocation moves to the target environment."""
        pass
