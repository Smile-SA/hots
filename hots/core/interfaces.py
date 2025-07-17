from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Any

class IngestionPlugin(ABC):
    @abstractmethod
    def load_initial(self) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        pass

    @abstractmethod
    def load_next(self) -> pd.DataFrame:
        pass

class ClusteringPlugin(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> pd.Series:
        pass

class OptimizationPlugin(ABC):
    @abstractmethod
    def solve(self, df_host: pd.DataFrame, labels: pd.Series) -> Any:
        pass

class HeuristicPlugin(ABC):
    @abstractmethod
    def adjust(self, solution: Any) -> Any:
        pass

class ConnectorPlugin(ABC):
    @abstractmethod
    def apply_moves(self, moves: Any) -> None:
        pass
