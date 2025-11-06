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
        self.pending_changes = {}

    def apply_moves(self, moves):
        """Publish moves (list of dicts) to Kafka."""
        for move in moves:
            # self.pending_changes = {
            #     'move': [{'container': c, 'node': n} for c, n in moves],
            #     tcol: str(current_time),
            # }
            # send JSON over Kafka topic
            msg = json.dumps(move).encode('utf-8')
            self.producer.produce(self.topic, msg)
        self.producer.flush()
        # print('Sending these moving containers (real-time):')
        # print(self.pending_changes)
        # it.kafka_producer.produce(
        #     it.kafka_topics['docker_replacer'],
        #     json.dumps(it.pending_changes),
        # )
        # it.kafka_producer.flush()
        # time.sleep(1)
