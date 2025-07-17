from core.interfaces import ConnectorPlugin
from confluent_kafka import Producer
import json

class KafkaConnector(ConnectorPlugin):
    def __init__(self, params, instance):
        self.producer = Producer({'bootstrap.servers': params['bootstrap.servers']})
        self.topic = params['topics'][0]

    def apply_moves(self, solution):
        moves = solution.extract_moves()  # assume your Model class offers this
        for m in moves:
            self.producer.produce(self.topic, json.dumps(m))
        self.producer.flush()
