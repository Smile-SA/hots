"""Unit tests for configuration loading and validation."""

import json

import pytest

from hots.config.loader import load_config

_BASE = {
    "clustering": {"method": "kmeans", "nb_clusters": 3, "parameters": {"nb_clusters": 3}},
    "optimization": {"backend": "pyomo", "parameters": {"solver": "glpk"}},
    "problem": {"type": "placement", "parameters": {}},
    "connector": {
        "type": "file",
        "parameters": {
            "data_folder": "./data",
            "tick_field": "timestamp",
            "individual_field": "container_id",
            "host_field": "machine_id",
            "metrics": ["cpu"],
        },
    },
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
    assert cfg.connector.parameters["metrics"] == ["cpu"]


def test_multi_metric_config_is_rejected(tmp_path):
    def two_metrics(cfg):
        cfg["connector"]["parameters"]["metrics"] = ["cpu", "mem"]

    with pytest.raises(ValueError, match="one metric"):
        load_config(_write(tmp_path, two_metrics))


def test_unknown_key_is_rejected(tmp_path):
    """extra='forbid' should catch typos in top-level keys."""

    def typo(cfg):
        cfg["problemm"] = cfg.pop("problem")

    with pytest.raises(ValueError):
        load_config(_write(tmp_path, typo))
