"""Configuration loader for the HOTS application."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class KafkaConfig:
    """Configuration for the Kafka connector: topics and optional schema settings."""

    topics: List[str]
    schema: Optional[Dict[str, Any]] = None
    schema_url: Optional[str] = None
    connector_url: str = ''


@dataclass
class ClusteringConfig:
    """Configuration for the clustering plugin: method name and its parameters."""

    method: str
    nb_clusters: int
    parameters: Dict[str, Any]


@dataclass
class OptimizationConfig:
    """Configuration for the optimization plugin: backend and its parameters."""

    backend: str
    parameters: Dict[str, Any]


@dataclass
class ProblemConfig:
    """Configuration for the domain problem plugin: type and its parameters."""

    type: str
    parameters: Dict[str, Any]


@dataclass
class ConnectorConfig:
    """Configuration for the connector plugin: type and its parameters."""

    type: str
    parameters: Dict[str, Any]


@dataclass
class LoggingConfig:
    """Configuration for application‑wide logging."""

    level: str               # e.g. "INFO" or "DEBUG"
    filename: Optional[str]  # if None, logs to stdout
    fmt: str                 # e.g. "%(asctime)s %(levelname)s: %(message)s"


@dataclass
class ReportingConfig:
    """Configuration for reporting: folders and file paths for outputs."""

    results_folder: Path
    metrics_file: Path
    plots_folder: Path


@dataclass
class AppConfig:
    """Top‑level application configuration, combining all sub‑configs."""

    data_folder: Path
    tick_field: str
    host_field: str
    individual_field: str
    metrics: List[str]
    time_limit: int
    kafka: Optional[KafkaConfig]
    clustering: ClusteringConfig
    optimization: OptimizationConfig
    problem: ProblemConfig
    connector: ConnectorConfig
    logging: LoggingConfig
    reporting: ReportingConfig


def load_config(path: Path) -> AppConfig:
    """
    Load JSON configuration from the given path into nested dataclasses.

    :param path: Path to the JSON config file.
    :return: An AppConfig instance populated from the file.
    """
    raw = json.loads(Path(path).read_text())

    kafka = None
    if raw.get('kafka') is not None:
        kafka = KafkaConfig(**raw['kafka'])

    clustering = ClusteringConfig(**raw['clustering'])
    optimization = OptimizationConfig(**raw['optimization'])
    problem = ProblemConfig(**raw['problem'])
    connector = ConnectorConfig(**raw['connector'])

    rpt = raw['reporting']
    logging_cfg = LoggingConfig(**raw['logging'])
    reporting = ReportingConfig(
        results_folder=Path(rpt['results_folder']),
        metrics_file=Path(rpt['metrics_file']),
        plots_folder=Path(rpt['plots_folder']),
    )

    return AppConfig(
        data_folder=Path(raw['data_folder']),
        tick_field=raw['tick_field'],
        host_field=raw['host_field'],
        individual_field=raw['individual_field'],
        metrics=raw['metrics'],
        time_limit=raw['time_limit'],
        kafka=kafka,
        clustering=clustering,
        optimization=optimization,
        problem=problem,
        connector=connector,
        logging=logging_cfg,
        reporting=reporting
    )
