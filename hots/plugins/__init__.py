# hots/plugins/__init__.py

# ingestion
from .ingestion.csv_reader import CSVReader
from .ingestion.kafka_reader import KafkaReader

# connector
from .connector.kafka_connector import KafkaConnector

# clustering
from .clustering.kmeans import StreamKMeans
from .clustering.hierarchical import HierarchicalClustering
from .clustering.spectral import SpectralClustering
from .clustering.custom_spectral import CustomSpectralClustering

# optimization
from .optimization.pyomo_model import PyomoModel

# heuristic
from .heuristic.spread import SpreadHeuristic


class ReaderFactory:
    @staticmethod
    def create(cfg, instance):
        t = cfg.type.lower()
        if t == 'csv':
            return CSVReader(cfg.parameters, instance)
        elif t == 'kafka':
            return KafkaReader(cfg.parameters, instance)
        else:
            raise ValueError(f"Unknown reader type: {cfg.type}")


class KafkaPlugin:
    @staticmethod
    def create_producer(cfg):
        from confluent_kafka import Producer
        return Producer({'bootstrap.servers': cfg.parameters['bootstrap.servers']})

    @staticmethod
    def create_consumer(cfg):
        from confluent_kafka import Consumer
        consumer = Consumer(cfg.parameters['consumer_conf'])
        consumer.subscribe(cfg.parameters['topics'])
        return consumer

    @staticmethod
    def clear_topics():
        # Implement topic/offset reset if your Kafka setup requires it
        pass


class ClusteringFactory:
    @staticmethod
    def create(cfg, instance):
        m = cfg.method.lower()
        if m in ('kmeans', 'stream_kmeans'):
            return StreamKMeans(cfg.parameters, instance)
        elif m == 'hierarchical':
            return HierarchicalClustering(cfg.parameters, instance)
        elif m == 'spectral':
            return SpectralClustering(cfg.parameters, instance)
        elif m == 'custom_spectral':
            return CustomSpectralClustering(cfg.parameters, instance)
        else:
            raise ValueError(f"Unknown clustering method: {cfg.method}")


class OptimizationFactory:
    @staticmethod
    def create(cfg, instance):
        s = cfg.solver.lower()
        if s == 'pyomo':
            return PyomoModel(cfg.parameters, instance)
        else:
            raise ValueError(f"Unknown optimization solver: {cfg.solver}")


class HeuristicFactory:
    @staticmethod
    def create(cfg, instance):
        t = cfg.type.lower()
        if t == 'spread':
            return SpreadHeuristic(cfg.parameters, instance)
        else:
            raise ValueError(f"Unknown heuristic type: {cfg.type}")


class ConnectorFactory:
    @staticmethod
    def create(cfg, instance):
        t = cfg.type.lower()
        if t == 'kafka':
            return KafkaConnector(cfg.parameters, instance)
        else:
            raise ValueError(f"Unknown connector type: {cfg.type}")
