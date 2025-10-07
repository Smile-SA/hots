"""HOTS plugin package."""

from .clustering.custom_spectral import CustomSpectralClustering
from .clustering.hierarchical import HierarchicalClustering
from .clustering.kmeans import StreamKMeans
from .clustering.spectral import SpectralClustering
from .connector.file_connector import FileConnector
from .connector.kafka_connector import KafkaConnector
from .ingestion.csv_reader import CSVReader
from .ingestion.kafka_reader import KafkaReader
from .optimization.pyomo_model import PyomoModel


class ReaderFactory:
    """Factory for data ingestion plugins."""

    @staticmethod
    def create(cfg, instance):
        """Create and return an ingestion plugin based on the config."""
        t = cfg.type.lower()
        if t == 'csv':
            return CSVReader(cfg.parameters, instance)
        if t == 'kafka':
            return KafkaReader(cfg.parameters, instance)
        raise ValueError(f'Unknown reader type: {cfg.type}')


class KafkaPlugin:
    """Utility to manage Kafka producers and consumers."""

    @staticmethod
    def create_producer(cfg):
        """Create and return a Kafka producer."""
        from confluent_kafka import Producer
        return Producer({'bootstrap.servers': cfg.parameters['bootstrap.servers']})

    @staticmethod
    def create_consumer(cfg):
        """Create and return a Kafka consumer subscribed to topics."""
        from confluent_kafka import Consumer
        consumer = Consumer(cfg.parameters['consumer_conf'])
        consumer.subscribe(cfg.parameters['topics'])
        return consumer

    @staticmethod
    def clear_topics():
        """Clear offsets for configured Kafka topics (if needed)."""
        pass


class ClusteringFactory:
    """Factory for clustering plugins."""

    @staticmethod
    def create(cfg, instance):
        """Create and return a clustering plugin based on config."""
        m = cfg.method.lower()
        if m in ('kmeans', 'stream_kmeans'):
            return StreamKMeans(cfg.parameters, instance)
        if m == 'hierarchical':
            return HierarchicalClustering(cfg.parameters, instance)
        if m == 'spectral':
            return SpectralClustering(cfg.parameters, instance)
        if m == 'custom_spectral':
            return CustomSpectralClustering(cfg.parameters, instance)
        raise ValueError(f'Unknown clustering method: {cfg.method}')


class OptimizationFactory:
    """Factory for optimization plugins."""

    @staticmethod
    def create(cfg, instance):
        """Create and return an optimization plugin based on config."""
        s = cfg.solver.lower()
        if s == 'pyomo':
            return PyomoModel(cfg.parameters, instance)
        raise ValueError(f'Unknown optimization solver: {cfg.solver}')


class ConnectorFactory:
    """Factory for connector plugins."""

    @staticmethod
    def create(cfg, instance):
        """Create and return a connector plugin based on config."""
        t = cfg.type.lower()
        if t == 'kafka':
            return KafkaConnector(cfg.parameters, instance)
        if t == 'file':
            return FileConnector(cfg.parameters)
        raise ValueError(f'Unknown connector type: {cfg.type}')
