# hots/plugins/connector/kafka_connector.py

"""Kafka connector plugin for HOTS."""

import json

from confluent_kafka import Producer

from hots.core.interfaces import ConnectorPlugin


class KafkaConnector(ConnectorPlugin):
    """Connector plugin that publishes moves to Kafka."""

    def __init__(self, params, instance):
        """Initialize Kafka producer and topic."""
        self.producer = Producer(
            {'bootstrap.servers': params['bootstrap.servers']}
        )
        self.topic = params['topics'][0]

    def apply_moves(self, solution):
        """Produce relocation moves to the configured Kafka topic."""
        moves = solution.extract_moves()
        for m in moves:
            self.producer.produce(self.topic, json.dumps(m))
        self.producer.flush()
