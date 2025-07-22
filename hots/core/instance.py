# hots/core/instance.py

"""HOTS core functionality: manage state and data ingestion."""

from pathlib import Path
from typing import Any, Dict, List

from hots.config.loader import AppConfig
from hots.plugins import KafkaPlugin, ReaderFactory
from hots.preprocessing.tools import build_df_from_containers

import pandas as pd


class Instance:
    """Maintain application state, including data, Kafka, and metrics."""

    def __init__(self, config: AppConfig):
        """Initialize the Instance with configuration and initial data."""
        self.config = config
        self.metrics_history: List[Dict[str, Any]] = []
        self.results_file: Path = config.reporting.metrics_file

        self.reader = ReaderFactory.create(config.reader, self)
        self.kafka_producer = None
        self.kafka_consumer = None
        if config.kafka:
            self.kafka_producer = KafkaPlugin.create_producer(config.kafka)
            self.kafka_consumer = KafkaPlugin.create_consumer(config.kafka)

        self.df_indiv, self.df_host, self.df_meta = self._load_initial_data()
        self.current_solution = None
        self.cluster_labels = None

    def _load_initial_data(self):
        """
        Load the raw individual-level data via the reader
        and derive the host-level DataFrame by aggregation.
        """
        df_indiv, _, df_meta = self.reader.load_initial()
        df_host = build_df_from_containers(
            df_indiv,
            tick_field=self.config.tick_field,
            host_field=self.config.host_field,
            individual_field=self.config.individual_field,
            metrics=self.config.metrics,
        )
        return df_indiv, df_host, df_meta

    @staticmethod
    def clear_kafka_topics() -> None:
        """Clear all configured Kafka topics."""
        KafkaPlugin.clear_topics()

    def update_data(self, new_df_indiv: pd.DataFrame) -> None:
        """Update the dataframes with newly ingested individual-level data."""
        self.df_indiv = pd.concat(
            [self.df_indiv, new_df_indiv],
            ignore_index=True,
        )
        self.df_host = build_df_from_containers(
            self.df_indiv,
            tick_field=self.config.tick_field,
            host_field=self.config.host_field,
            individual_field=self.config.individual_field,
            metrics=self.config.metrics,
        )

    def get_id_map(self) -> Dict[Any, int]:
        """Get a mapping from container IDs to integer indices."""
        unique = sorted(
            self.df_indiv[self.config.individual_field].unique()
        )
        return {cid: idx for idx, cid in enumerate(unique)}
