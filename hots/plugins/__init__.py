"""HOTS plugin package."""

import importlib

from .clustering.custom_spectral import CustomSpectralClustering
from .clustering.hierarchical import HierarchicalClustering
from .clustering.kmeans import StreamKMeans
from .clustering.spectral import SpectralClustering
from .connector.file_connector import FileConnector
from .optimization.factory import OptimizationFactory


class KafkaPlugin:
    """Utility to manage Kafka producers and consumers."""

    @staticmethod
    def create_producer(cfg):
        """Create and return a Kafka producer."""
        from confluent_kafka import Producer

        return Producer({"bootstrap.servers": cfg.parameters["bootstrap.servers"]})

    @staticmethod
    def create_consumer(cfg):
        """Create and return a Kafka consumer subscribed to topics."""
        from confluent_kafka import Consumer

        consumer = Consumer(cfg.parameters["consumer_conf"])
        consumer.subscribe(cfg.parameters["topics"])
        return consumer


class ClusteringFactory:
    """Factory for clustering plugins."""

    @staticmethod
    def create(cfg, instance):
        """Create and return a clustering plugin based on config."""
        m = cfg.method.lower()
        if m in ("kmeans", "stream_kmeans"):
            return StreamKMeans(cfg.parameters, instance)
        if m == "hierarchical":
            return HierarchicalClustering(cfg.parameters, instance)
        if m == "spectral":
            return SpectralClustering(cfg.parameters, instance)
        if m == "custom_spectral":
            return CustomSpectralClustering(cfg.parameters, instance)
        raise ValueError(f"Unknown clustering method: {cfg.method}")


class ConnectorFactory:
    """Factory for connector plugins."""

    @staticmethod
    def create(cfg, instance):
        """Create and return a connector plugin based on config."""
        t = cfg.type.lower()
        if t == "kafka":
            try:
                from hots.plugins.connector.kafka_connector import KafkaConnector
            except ImportError as e:
                raise ImportError(
                    "Kafka connector requires the optional 'kafka' extra. "
                    "Install with: uv sync --extra kafka (or pip install .[kafka])."
                ) from e
            return KafkaConnector(cfg.parameters)
        if t == "file":
            return FileConnector(cfg.parameters)
        raise ValueError(f"Unknown connector type: {cfg.type}")


class ProblemFactory:
    """Factory for domain problem plugins."""

    # Explicit registry of built-in problem types -> "module:ClassName".
    _REGISTRY: dict[str, str] = {
        "placement": "hots.plugins.problem.placement:PlacementPlugin",
    }

    @classmethod
    def register(cls, name: str, target: str) -> None:
        """Register a problem plugin as ``"package.module:ClassName"``."""
        cls._REGISTRY[name.lower()] = target

    @classmethod
    def create(cls, cfg, instance):
        """Create and return a problem plugin based on config."""
        key = cfg.type.lower()
        target = cls._REGISTRY.get(key)
        if target is None:
            raise ValueError(f"Unknown problem type: {cfg.type!r} (known: {sorted(cls._REGISTRY)})")
        module_path, cls_name = target.split(":")
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)(instance)


__all__ = [
    "ClusteringFactory",
    "ConnectorFactory",
    "FileConnector",
    "KafkaPlugin",
    "OptimizationFactory",
    "ProblemFactory",
    "CustomSpectralClustering",
    "HierarchicalClustering",
    "SpectralClustering",
    "StreamKMeans",
]
