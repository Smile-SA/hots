from pathlib import Path
import json
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class KafkaConfig(BaseModel):
    topics: List[str]
    schema: Optional[Dict[str, Any]] = None
    schema_url: Optional[str] = None
    connector_url: str

class ReaderConfig(BaseModel):
    type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ClusteringConfig(BaseModel):
    method: str
    nb_clusters: int
    parameters: Dict[str, Any] = Field(default_factory=dict)

class OptimizationConfig(BaseModel):
    solver: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class HeuristicConfig(BaseModel):
    type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ConnectorConfig(BaseModel):
    type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ReportingConfig(BaseModel):
    results_folder: Path
    metrics_file: Path
    plots_folder: Path

class AppConfig(BaseModel):
    data_folder: Path
    tick_field: str
    host_field: str
    individual_field: str
    metrics: List[str]
    reader: ReaderConfig
    kafka: Optional[KafkaConfig] = None
    clustering: ClusteringConfig
    optimization: OptimizationConfig
    heuristic: HeuristicConfig
    connector: ConnectorConfig
    reporting: ReportingConfig

def load_config(path: Path) -> AppConfig:
    raw = json.loads(Path(path).read_text())
    return AppConfig(**raw)
