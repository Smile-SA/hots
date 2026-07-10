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
        self._known_hosts: set[Any] = set()

    def _load_initial_data(self, connector):
        """
        Load the raw individual-level data via the reader
        and derive the host-level DataFrame by aggregation.
        """
        params = self.config.connector.parameters
        self.df_indiv, _, self.df_meta = connector.load_initial()
        self._known_hosts = set(self.df_indiv[params.get("host_field")].unique())
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

    @property
    def history_ticks(self) -> int | None:
        """Number of trailing ticks retained in memory, or None to keep everything.

        Defaults to the working window plus one tick increment of slack, which is
        the most `get_working_df()` can ever ask for.
        """
        params = self.config.connector.parameters
        configured = params.get("history_ticks")
        if configured is not None:
            return int(configured)

        window_duration = params.get("window_duration")
        if window_duration is None:
            return None
        return int(window_duration) + int(params.get("tick_increment", 1)) + 1

    @staticmethod
    def clear_kafka_topics() -> None:
        """Clear all configured Kafka topics."""
        KafkaPlugin.clear_topics()

    def _invalidate_caches(self) -> None:
        """Mark derived structures as needing a refresh on next access."""
        self._id_map_stale = True

    def update_data(self, new_df_indiv: pd.DataFrame) -> None:
        """Update the dataframes with newly ingested individual-level data.

        The host aggregate is recomputed only for the ticks the new batch
        touches, and both frames are trimmed to `history_ticks`, so memory and
        per-loop cost stay bounded over a long online run.
        """
        if new_df_indiv is None or new_df_indiv.empty:
            return

        params = self.config.connector.parameters
        tick_field = params.get("tick_field")
        host_field = params.get("host_field")
        metrics = params.get("metrics")

        self.df_indiv = pd.concat([self.df_indiv, new_df_indiv], ignore_index=True)
        self._known_hosts.update(new_df_indiv[host_field].unique())

        # A batch may complete a tick already partially aggregated (Kafka batches
        # do not align on tick boundaries), so re-aggregate whole affected ticks.
        touched = new_df_indiv[tick_field].unique()
        affected = self.df_indiv[self.df_indiv[tick_field].isin(touched)]
        recomputed = build_df_from_containers(
            affected,
            tick_field=tick_field,
            host_field=host_field,
            metrics=metrics,
            hosts=sorted(self._known_hosts),
        )
        self.df_host = pd.concat(
            [self.df_host[~self.df_host[tick_field].isin(touched)], recomputed],
            ignore_index=True,
        ).sort_values([tick_field, host_field], ignore_index=True)

        self._prune_history()
        self._invalidate_caches()

    def _prune_history(self) -> None:
        """Drop rows older than the retention window from both frames."""
        retention = self.history_ticks
        if retention is None:
            return

        tick_field = self.config.connector.parameters.get("tick_field")
        cutoff = self.df_indiv[tick_field].max() - retention + 1
        self.df_indiv = self.df_indiv[self.df_indiv[tick_field] >= cutoff].reset_index(drop=True)
        self.df_host = self.df_host[self.df_host[tick_field] >= cutoff].reset_index(drop=True)

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
