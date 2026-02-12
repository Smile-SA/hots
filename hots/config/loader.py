"""Configuration loader for the HOTS application."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Base model config


class _Cfg(BaseModel):
    """
    Shared configuration for all config models.

    - extra="forbid" catches typos / unknown keys early.
    - validate_assignment is handy if configs are modified programmatically.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# Core configuration sections


class KafkaConfig(_Cfg):
    """Configuration for the Kafka connector: topics and optional schema settings."""

    topics: list[str]
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    schema_url: str | None = None
    connector_url: str = ""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ClusteringConfig(_Cfg):
    """Configuration for the clustering plugin: method name and its parameters."""

    method: str
    nb_clusters: int
    parameters: dict[str, Any]


class OptimizationConfig(_Cfg):
    """Configuration for the optimization plugin: backend and its parameters."""

    backend: str
    parameters: dict[str, Any]


class ProblemConfig(_Cfg):
    """Configuration for the domain problem plugin: type and its parameters."""

    type: str
    parameters: dict[str, Any]


class ConnectorConfig(_Cfg):
    """Configuration for the connector plugin: type and its parameters."""

    type: str
    parameters: dict[str, Any]


class LoggingConfig(_Cfg):
    """Configuration for application-wide logging."""

    level: str  # e.g. "INFO" or "DEBUG"
    filename: str | None = None  # if None, logs to stdout
    fmt: str  # e.g. "%(asctime)s %(levelname)s: %(message)s"


class ReportingConfig(_Cfg):
    """Configuration for reporting: folders and file paths for outputs."""

    results_folder: Path
    metrics_file: Path
    plots_folder: Path


# Visualization configuration


class SaveConfig(_Cfg):
    enabled: bool = True
    plots_folder: Path | None = None  # if None -> don't save unless explicit out_path is passed
    default_format: str = "png"
    dpi: int = 150
    bbox_inches: str = "tight"
    close_on_save: bool = True

    @field_validator("plots_folder", mode="before")
    @classmethod
    def _empty_string_to_none(cls, v):
        # Preserve your previous behavior: "" (or None) means "no folder".
        if v in (None, ""):
            return None
        return v


class ColorConfig(_Cfg):
    clusters: Sequence[str] = (
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "turquoise",
        "chocolate",
        "navy",
        "lightcoral",
        "violet",
    )
    highlight: Sequence[str] = ("violet", "lightcoral", "navy", "chocolate", "turquoise")


class DendrogramConfig(_Cfg):
    leaf_rotation: float = 90.0
    leaf_font_size: float = 8.0


class SpringLayoutConfig(_Cfg):
    k: float = 0.15
    iterations: int = 20


class GraphConfig(_Cfg):
    spring_layout: SpringLayoutConfig = Field(default_factory=SpringLayoutConfig)
    show_edge_weights: bool = True


class NodesConfig(_Cfg):
    capacity_margin_ratio: float = 0.2
    sep_time_color: str = "red"
    sep_time_style: str = "--"
    capacity_color: str = "red"


class LiveConfig(_Cfg):
    pause_seconds: float = 0.5


class NamingConfig(_Cfg):
    prefix: str = ""
    suffix: str = ""
    slugify: bool = True


class VisualizationConfig(_Cfg):
    enabled: bool = True
    default_metric: str = "cpu"

    save: SaveConfig = Field(default_factory=SaveConfig)
    colors: ColorConfig = Field(default_factory=ColorConfig)

    dendrogram: DendrogramConfig = Field(default_factory=DendrogramConfig)
    graphs: GraphConfig = Field(default_factory=GraphConfig)
    nodes: NodesConfig = Field(default_factory=NodesConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)
    naming: NamingConfig = Field(default_factory=NamingConfig)

    @model_validator(mode="before")
    @classmethod
    def _unwrap_optional_visualization_root(cls, data):
        """
        Preserve old tolerance:
        - accept either {"enabled": ..., ...}
        - or {"visualization": {...}} as an extra wrapper
        """
        if isinstance(data, Mapping) and "visualization" in data and len(data) == 1:
            inner = data.get("visualization")
            if isinstance(inner, Mapping):
                return inner
        return data


# Top-level configuration


class AppConfig(_Cfg):
    """Top-level application configuration, combining all sub-configs."""

    time_limit: int | None = Field(default=None, ge=0)
    kafka: KafkaConfig | None = None
    clustering: ClusteringConfig
    optimization: OptimizationConfig
    problem: ProblemConfig
    connector: ConnectorConfig
    logging: LoggingConfig
    reporting: ReportingConfig
    viz: VisualizationConfig = Field(default_factory=VisualizationConfig, alias="visualization")

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,  # allows using either "viz" or "visualization" when needed
    )


def load_config(path: Path) -> AppConfig:
    """
    Load JSON configuration from the given path into nested validated models.

    :param path: Path to the JSON config file.
    :return: An AppConfig instance populated from the file.
    :raises ValueError: if the config is invalid.
    """
    raw_text = Path(path).read_text(encoding="utf-8")

    try:
        # Validate directly from JSON and apply aliases (visualization -> viz).
        return AppConfig.model_validate_json(raw_text)
    except ValidationError as exc:
        raise ValueError(f"Invalid config file at {path}:\n{exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file at {path}:\n{exc}") from exc
