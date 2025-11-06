# hots/plugins/connector/kafka_connector.py

"""Kafka connector plugin for HOTS."""

import json

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

from hots.core.interfaces import ConnectorPlugin

import pandas as pd


class KafkaConnector(ConnectorPlugin):
    """Connector plugin that read data from Kafka and publishes moves to it."""

    def __init__(self, params):
        """Initialize Kafka producer and topic."""
        self.consumer = Consumer(params['consumer_conf'])
        self.consumer.subscribe(params['topics'])
        self.tick_field = params['tick_field']
        self.indiv_field = params['individual_field']
        self.host_field = params['host_field']
        self.metrics = params['metrics']
        self.batch_size = params.get('batch_size', 1)

        self.producer = Producer(
            {'bootstrap.servers': params['bootstrap.servers']}
        )
        self.topic = params['topics'][0]
        self.pending_changes = {}

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
