"""Unit tests for configuration loading and validation."""

import json

import pytest

from hots.config.loader import load_config

_BASE = {
    "data": {
        "tick_field": "timestamp",
        "individual_field": "container_id",
        "host_field": "machine_id",
        "metrics": ["cpu"],
    },
    "clustering": {"method": "kmeans", "nb_clusters": 3, "parameters": {"nb_clusters": 3}},
    "optimization": {"backend": "pyomo", "parameters": {"solver": "glpk"}},
    "problem": {"type": "placement", "parameters": {}},
    "connector": {"type": "file", "parameters": {"data_folder": "./data"}},
    "logging": {"level": "INFO", "fmt": "%(message)s"},
    "reporting": {
        "results_folder": "./out",
        "metrics_file": "./out/metrics.csv",
        "plots_folder": "./out/plots",
    },
}


def _write(tmp_path, overrides=None):
    cfg = json.loads(json.dumps(_BASE))
    if overrides:
        overrides(cfg)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_single_metric_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert cfg.data.metrics == ["cpu"]


def test_multi_metric_config_is_rejected(tmp_path):
    def two_metrics(cfg):
        cfg["data"]["metrics"] = ["cpu", "mem"]

    with pytest.raises(ValueError, match="one metric"):
        load_config(_write(tmp_path, two_metrics))


def test_data_section_is_mirrored_into_connector_parameters(tmp_path):
    """Connector plugins read their params dict, so fields must reach it."""
    cfg = load_config(_write(tmp_path))
    params = cfg.connector.parameters

    assert params["tick_field"] == "timestamp"
    assert params["individual_field"] == "container_id"
    assert params["host_field"] == "machine_id"
    assert params["metrics"] == ["cpu"]


def test_legacy_config_without_data_section_still_loads(tmp_path):
    """Fields declared the old way (under connector.parameters) populate `data`."""

    def legacy(cfg):
        params = cfg["connector"]["parameters"]
        params.update(cfg.pop("data"))

    cfg = load_config(_write(tmp_path, legacy))

    assert cfg.data.tick_field == "timestamp"
    assert cfg.data.individual_field == "container_id"
    assert cfg.data.host_field == "machine_id"
    assert cfg.data.metrics == ["cpu"]


def test_legacy_multi_metric_is_still_rejected(tmp_path):
    """The single-metric rule must apply to the legacy spelling too."""

    def legacy_two_metrics(cfg):
        params = cfg["connector"]["parameters"]
        params.update(cfg.pop("data"))
        params["metrics"] = ["cpu", "mem"]

    with pytest.raises(ValueError, match="one metric"):
        load_config(_write(tmp_path, legacy_two_metrics))


def test_conflicting_declarations_are_rejected(tmp_path):
    """Declaring a field in both places with different values is an error."""

    def conflict(cfg):
        cfg["connector"]["parameters"]["host_field"] = "node_id"  # data says machine_id

    with pytest.raises(ValueError, match="Conflicting values"):
        load_config(_write(tmp_path, conflict))


def test_matching_declarations_in_both_places_are_allowed(tmp_path):
    """Redundant but consistent declarations are fine."""

    def duplicate(cfg):
        cfg["connector"]["parameters"]["host_field"] = "machine_id"

    cfg = load_config(_write(tmp_path, duplicate))
    assert cfg.data.host_field == "machine_id"


def test_missing_metrics_is_rejected(tmp_path):
    def drop_metrics(cfg):
        del cfg["data"]["metrics"]

    with pytest.raises(ValueError):
        load_config(_write(tmp_path, drop_metrics))


def test_unknown_key_is_rejected(tmp_path):
    """extra='forbid' should catch typos in top-level keys."""

    def typo(cfg):
        cfg["problemm"] = cfg.pop("problem")

    with pytest.raises(ValueError):
        load_config(_write(tmp_path, typo))
