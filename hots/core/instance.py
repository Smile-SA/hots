# hots/core/instance.py

"""HOTS core functionality: manage state and data ingestion."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from hots.config.loader import AppConfig
from hots.plugins import KafkaPlugin
from hots.utils.tools import build_df_from_containers


@dataclass(frozen=True)
class FieldNames:
    """Column names used across the project."""

    tick: str
    host: str
    individual: str


class Instance:
    """Maintain application state, including data, Kafka, and metrics."""

    def __init__(self, config: AppConfig):
        """Initialize the Instance with configuration and initial data."""
        self.config = config
        self.metrics_history: list[dict[str, Any]] = []

        self.kafka_producer = None
        self.kafka_consumer = None
        if config.kafka:
            self.kafka_producer = KafkaPlugin.create_producer(config.kafka)
            self.kafka_consumer = KafkaPlugin.create_consumer(config.kafka)

        self.df_indiv, self.df_host, self.df_meta = None, None, None
        # Append-only: an ID keeps its index for the whole run, so matrices and
        # label arrays stay positionally comparable across windows.
        self._id_map: dict[Any, int] = {}
        self._inv_id_map: dict[int, Any] = {}
        self._id_map_stale = True
        self.current_solution = None
        self.cluster_labels = None
        self.fields = None

    def _load_initial_data(self, connector):
        """
        Load the raw individual-level data via the reader
        and derive the host-level DataFrame by aggregation.
        """
        params = self.config.connector.parameters
        self.df_indiv, _, self.df_meta = connector.load_initial()
        self.df_host = build_df_from_containers(
            self.df_indiv,
            tick_field=params.get("tick_field"),
            host_field=params.get("host_field"),
            metrics=params.get("metrics"),
        )
        self.fields = FieldNames(
            tick=params.get("tick_field"),
            host=params.get("host_field"),
            individual=params.get("individual_field"),
        )
        self._invalidate_caches()

    @staticmethod
    def clear_kafka_topics() -> None:
        """Clear all configured Kafka topics."""
        KafkaPlugin.clear_topics()

    def _invalidate_caches(self) -> None:
        """Mark derived structures as needing a refresh on next access."""
        self._id_map_stale = True

    def update_data(self, new_df_indiv: pd.DataFrame) -> None:
        """Update the dataframes with newly ingested individual-level data."""
        params = self.config.connector.parameters
        self.df_indiv = pd.concat(
            [self.df_indiv, new_df_indiv],
            ignore_index=True,
        )
        self.df_host = build_df_from_containers(
            self.df_indiv,
            tick_field=params.get("tick_field"),
            host_field=params.get("host_field"),
            metrics=params.get("metrics"),
        )
        self._invalidate_caches()

    def get_id_map(self) -> dict[Any, int]:
        """Get a mapping from container IDs to integer indices.

        Indices are assigned append-only: an ID seen once keeps its index for the
        rest of the run, and IDs first seen later take the next free indices. A
        container arriving with an ID that sorts before existing ones therefore
        never shifts the indices the matrices and label arrays are keyed on.
        """
        if self._id_map_stale:
            indiv_field = self.config.connector.parameters.get("individual_field")
            unseen = set(self.df_indiv[indiv_field].unique()) - self._id_map.keys()
            # sorted() only to keep index assignment deterministic within a batch
            for cid in sorted(unseen):
                idx = len(self._id_map)
                self._id_map[cid] = idx
                self._inv_id_map[idx] = cid
            self._id_map_stale = False
        return self._id_map

    def get_inv_id_map(self) -> dict[int, Any]:
        """Get the inverse mapping from indices to container IDs."""
        self.get_id_map()
        return self._inv_id_map

    def get_working_df(self, tmin, tmax, inclusive=True):
        """Get data for the current time window."""
        df = self.df_indiv
        tick_field = self.config.connector.parameters.get("tick_field")
        if inclusive:
            mask = (df[tick_field] >= tmin) & (df[tick_field] <= tmax)
        else:
            mask = (df[tick_field] > tmin) & (df[tick_field] < tmax)
        return df.loc[mask]
