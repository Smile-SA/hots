"""Contract tests: concrete plugins must satisfy their widened ABCs."""

import inspect

import pytest

from hots.core.interfaces import (
    ClusteringPlugin,
    ConnectorPlugin,
    OptimizationPlugin,
    ProblemPlugin,
)
from hots.plugins import ProblemFactory
from hots.plugins.clustering.custom_spectral import CustomSpectralClustering
from hots.plugins.clustering.hierarchical import HierarchicalClustering
from hots.plugins.clustering.kmeans import StreamKMeans
from hots.plugins.clustering.spectral import SpectralClustering
from hots.plugins.connector.file_connector import FileConnector
from hots.plugins.optimization.pyomo_model import PyomoModel
from hots.plugins.problem.placement import PlacementPlugin

CLUSTERING_PLUGINS = [
    StreamKMeans,
    HierarchicalClustering,
    SpectralClustering,
    CustomSpectralClustering,
]


def _abstract_methods(abc_cls) -> set[str]:
    return {
        name
        for name, member in inspect.getmembers(abc_cls)
        if getattr(member, "__isabstractmethod__", False)
    }


@pytest.mark.parametrize("plugin", CLUSTERING_PLUGINS)
def test_clustering_plugins_are_subclasses(plugin):
    assert issubclass(plugin, ClusteringPlugin)
    # Every abstract method is overridden (no leftover abstractness)
    assert not getattr(plugin, "__abstractmethods__", set())


def test_optimization_plugin_implements_full_surface():
    assert issubclass(PyomoModel, OptimizationPlugin)
    assert not PyomoModel.__abstractmethods__
    for name in _abstract_methods(OptimizationPlugin):
        assert callable(getattr(PyomoModel, name)), name


def test_connector_plugin_implements_full_surface():
    assert issubclass(FileConnector, ConnectorPlugin)
    assert not FileConnector.__abstractmethods__
    for name in ("load_initial", "load_next", "apply_moves", "close"):
        assert callable(getattr(FileConnector, name))


def test_problem_plugin_implements_full_surface():
    assert issubclass(PlacementPlugin, ProblemPlugin)
    assert not PlacementPlugin.__abstractmethods__
    for name in ("adjust", "initial", "build_place_adj_matrix"):
        assert callable(getattr(PlacementPlugin, name))


def test_problem_factory_resolves_placement():
    class _Cfg:
        type = "placement"

    # PlacementPlugin only touches instance.config.connector.parameters at init
    class _ConnCfg:
        parameters = {
            "individual_field": "container_id",
            "host_field": "machine_id",
            "tick_field": "timestamp",
            "metrics": ["cpu"],
        }

    class _ConfigNS:
        connector = _ConnCfg()

    class _Instance:
        config = _ConfigNS()

    plugin = ProblemFactory.create(_Cfg(), _Instance())
    assert isinstance(plugin, PlacementPlugin)


def test_problem_factory_rejects_unknown_type():
    class _Cfg:
        type = "does_not_exist"

    with pytest.raises(ValueError, match="Unknown problem type"):
        ProblemFactory.create(_Cfg(), object())


def test_clustering_abc_initializes_pipeline_attributes():
    class _Dummy(ClusteringPlugin):
        def fit(self, df):
            return df

    d = _Dummy()
    assert d.labels is None
    assert d.u_mat is None
    assert d.w_mat is None
    assert d.clust_mat is None
    assert d.profiles is None
