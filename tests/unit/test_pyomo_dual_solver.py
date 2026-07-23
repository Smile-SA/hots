"""Unit tests for PyomoModel's dual-reporting solver checks.

Conflict detection (see evaluator.get_moving_containers_*) is driven entirely
by LP shadow prices extracted in ``fill_dual_values``. That extraction is
documented to depend on the solver actually populating Pyomo's ``dual``
Suffix (verified for GLPK only); these tests cover the two places that now
surface a departure from that assumption instead of silently handing back an
empty/all-zero conflict graph.
"""

import logging

from hots.core.instance import FieldNames
from hots.plugins.optimization.pyomo_model import PyomoModel

TICK, INDIV, HOST, METRIC = "timestamp", "container_id", "machine_id", "cpu"


class _ClusteringConfig:
    nb_clusters = 2


class _Config:
    clustering = _ClusteringConfig()


class _Instance:
    """Minimal stand-in: PyomoModel.__init__ only reads these attributes."""

    fields = FieldNames(tick=TICK, host=HOST, individual=INDIV, metrics=(METRIC,))
    config = _Config()

    def __init__(self):
        self.df_indiv = None
        self.df_meta = None

    def get_id_map(self):
        return {"c_1": 0, "c_2": 1}


def test_verified_solver_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        PyomoModel({"solver": "glpk"}, _Instance())
    assert not caplog.records


def test_unverified_solver_warns(caplog):
    with caplog.at_level(logging.WARNING):
        PyomoModel({"solver": "cbc"}, _Instance())
    assert any("cbc" in r.getMessage() for r in caplog.records)


class _FakeConstraint:
    """Stand-in for a Pyomo IndexedConstraint: iterable indices + a name."""

    def __init__(self, indices, name):
        self._indices = list(indices)
        self.name = name

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return (self.name, idx)


class _FakeInstanceModel:
    def __init__(self, con, dual):
        self.must_link_c = con
        self.dual = dual


def _model_with_fake_solve(con, dual):
    model = PyomoModel({"solver": "glpk"}, _Instance())
    model.instance_model = _FakeInstanceModel(con, dual)
    return model


def test_fill_dual_values_reads_present_duals(caplog):
    con = _FakeConstraint([("c_1", "c_2")], "must_link_c")
    # Keyed by the ConstraintData object (con[idx]), like a real Pyomo Suffix
    # -- not by the raw index -- so this also guards against checking `idx in
    # dual` instead of `con[idx] in dual` when deciding whether to warn.
    dual = {con[("c_1", "c_2")]: 0.5}
    model = _model_with_fake_solve(con, dual)

    with caplog.at_level(logging.WARNING):
        model.fill_dual_values()

    assert model.last_duals == {("c_1", "c_2"): 0.5}
    assert not caplog.records


def test_fill_dual_values_warns_when_solver_reports_nothing(caplog):
    con = _FakeConstraint([("c_1", "c_2")], "must_link_c")
    model = _model_with_fake_solve(con, {})  # solver populated no entries at all

    with caplog.at_level(logging.WARNING):
        model.fill_dual_values()

    # Still returns a usable (zero-filled) mapping, but loudly, not silently.
    assert model.last_duals == {("c_1", "c_2"): 0.0}
    assert any("no dual values" in r.getMessage() for r in caplog.records)


def test_fill_dual_values_no_constraints_is_not_a_warning(caplog):
    con = _FakeConstraint([], "must_link_c")
    model = _model_with_fake_solve(con, {})

    with caplog.at_level(logging.WARNING):
        model.fill_dual_values()

    assert model.last_duals == {}
    assert not caplog.records
