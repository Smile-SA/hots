# config/loader.py

from pathlib import Path
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class KafkaConfig:
    topics: List[str]
    schema: Optional[Dict[str, Any]] = None
    schema_url: Optional[str] = None
    connector_url: str = ''


@dataclass
class ReaderConfig:
    type: str
    parameters: Dict[str, Any]


@dataclass
class ClusteringConfig:
    method: str
    nb_clusters: int
    parameters: Dict[str, Any]


@dataclass
class OptimizationConfig:
    solver: str
    parameters: Dict[str, Any]


@dataclass
class HeuristicConfig:
    type: str
    parameters: Dict[str, Any]


@dataclass
class ConnectorConfig:
    type: str
    parameters: Dict[str, Any]


@dataclass
class ReportingConfig:
    results_folder: Path
    metrics_file: Path
    plots_folder: Path


@dataclass
class AppConfig:
    data_folder: Path
    tick_field: str
    host_field: str
    individual_field: str
    metrics: List[str]
    reader: ReaderConfig
    kafka: Optional[KafkaConfig]
    clustering: ClusteringConfig
    optimization: OptimizationConfig
    heuristic: HeuristicConfig
    connector: ConnectorConfig
    reporting: ReportingConfig


def load_config(path: Path) -> AppConfig:
    """Load JSON config into nested dataclasses."""
    raw = json.loads(Path(path).read_text())

    reader = ReaderConfig(**raw['reader'])

    kafka = None
    if raw.get('kafka') is not None:
        kafka = KafkaConfig(**raw['kafka'])

    clustering = ClusteringConfig(**raw['clustering'])
    optimization = OptimizationConfig(**raw['optimization'])
    heuristic = HeuristicConfig(**raw['heuristic'])
    connector = ConnectorConfig(**raw['connector'])

    rpt = raw['reporting']
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
        reader=reader,
        kafka=kafka,
        clustering=clustering,
        optimization=optimization,
        heuristic=heuristic,
        connector=connector,
        reporting=reporting,
    )
