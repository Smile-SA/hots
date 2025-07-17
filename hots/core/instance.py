from pathlib import Path
import pandas as pd
from typing import Optional, Any, Tuple
from config.loader import AppConfig
from preprocessing.tools import build_df_from_containers
from plugins import ReaderFactory, KafkaPlugin, CSVReader

class Instance:
    def __init__(self, config: AppConfig):
        self.config = config
        self.reader = ReaderFactory.create(config.reader, self)
        self.kafka_producer = None
        self.kafka_consumer = None
        if config.kafka:
            self.kafka_producer = KafkaPlugin.create_producer(config.kafka)
            self.kafka_consumer = KafkaPlugin.create_consumer(config.kafka)

        self.df_indiv, self.df_host, self.df_meta = self._load_initial_data()
        self.current_solution: Optional[Any] = None
        self.cluster_labels: Optional[pd.Series] = None

    def _load_initial_data(self):
        """
        Always load the raw individual‐level data via the reader,
        then derive the host‐level DataFrame by aggregation.
        """
        # 1) Ingest raw container‐level ticks
        df_indiv, _, df_meta = self.reader.load_initial()

        # 2) Derive host‐level time series from the raw data
        df_host = build_df_from_containers(
            df_indiv,
            tick_field   = self.config.tick_field,
            host_field   = self.config.host_field,
            individual_field = self.config.individual_field,
            metrics      = self.config.metrics
        )

        return df_indiv, df_host, df_meta

    @staticmethod
    def clear_kafka_topics() -> None:
        KafkaPlugin.clear_topics()

    def update_data(self, new_df_indiv: pd.DataFrame) -> None:
        self.df_indiv = pd.concat([self.df_indiv, new_df_indiv], ignore_index=True)
        self.df_host = build_df_from_containers(
            self.df_indiv,
            tick_field=self.config.tick_field,
            host_field=self.config.host_field,
            individual_field=self.config.individual_field,
            metrics=self.config.metrics,
        )

    def get_id_map(self) -> dict:
        # Map container IDs to integer indices for clustering
        unique = sorted(self.df_indiv[self.config.individual_field].unique())
        return {cid: idx for idx, cid in enumerate(unique)}
