"""Kafka‑based ingestion plugin for HOTS."""

import json

from confluent_kafka import Consumer, KafkaError, KafkaException

from hots.core.interfaces import IngestionPlugin

import pandas as pd


class KafkaReader(IngestionPlugin):
    """Ingestion plugin that reads data from Kafka topics."""

    def __init__(self, parameters: dict, instance):
        """Initialize consumer and field mappings."""
        self.consumer = Consumer(parameters['consumer_conf'])
        self.consumer.subscribe(parameters['topics'])
        self.tick_field = parameters['tick_field']
        self.indiv_field = parameters['individual_field']
        self.host_field = parameters['host_field']
        self.metrics = parameters['metrics']
        self.batch_size = parameters.get('batch_size', 1)

    def load_initial(self) -> pd.DataFrame:
        """Load initial data batch (no‑op, returns empty)."""
        return pd.DataFrame(), None, None

    def load_next(self) -> pd.DataFrame:
        """Poll Kafka and return the next DataFrame of container records."""
        records = []
        for _ in range(self.batch_size):
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(msg.error())

            rec = json.loads(msg.value())
            for c in rec.get('containers', []):
                records.append({
                    self.tick_field: c['timestamp'],
                    self.indiv_field: c['container_id'],
                    self.host_field: c['machine_id'],
                    self.metrics[0]: c[self.metrics[0]],
                })

        if not records:
            return None
        return pd.DataFrame.from_records(records)
