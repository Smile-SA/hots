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
        """Publish moves (list of dicts) to Kafka."""
        if isinstance(solution, list):
            moves = solution
        else:
            moves = solution.extract_moves()

        for move in moves:
            # send JSON over Kafka topic
            msg = json.dumps(move).encode('utf-8')
            self.producer.produce(self.topic, msg)
        self.producer.flush()
