"""Regression tests for evaluator.py's persistent-model reuse.

On a no-new-container tick, evaluator.eval_solutions no longer calls
PyomoModel.build() again; it mutates the already-instantiated model in place
via update_w/update_sol_u/update_obj_place/update_adjacency_constraints (see
the comments at those call sites). These tests check that mutating an
existing model to a target (u, w, v, dv) state produces the same Param values
and the same solved objective as building fresh straight to that state --
i.e. that the incremental path is not silently dropping or mis-writing data.

They use real Pyomo + GLPK on tiny, deliberately non-degenerate instances
(distinguishable weights, no symmetry) so the LP has a unique optimum and the
comparison isn't confounded by the tied-dual-solution sensitivity documented
on PyomoModel.fill_dual_values.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from hots.core.instance import FieldNames

_pyomo_installed = importlib.util.find_spec("pyomo.environ") is not None
pytestmark = pytest.mark.skipif(not _pyomo_installed, reason="pyomo.environ (Pyomo) not installed")

if _pyomo_installed:
    import pyomo.environ  # noqa: F401
    from pyomo import environ as pe
    from pyomo.opt import SolverFactory

    from hots.plugins.optimization.pyomo_model import PyomoModel


def _skip_if_no_glpk():
    if not SolverFactory("glpk").available(False):
        pytest.skip("glpk not available on this runner")


class _ClusteringConfig:
    nb_clusters = 2


class _Config:
    clustering = _ClusteringConfig()


class _Instance:
    fields = FieldNames(tick="t", host="h", individual="c", metrics=("cpu",))
    config = _Config()

    def __init__(self, df_indiv, df_meta=None):
        self.df_indiv = df_indiv
        self.df_meta = df_meta

    def get_id_map(self):
        return {"c_1": 0, "c_2": 1, "c_3": 2}


def _assert_params_match(model_a, model_b, *param_names):
    for name in param_names:
        pa, pb = getattr(model_a.instance_model, name), getattr(model_b.instance_model, name)
        for i in model_a.id_list:
            for j in model_a.id_list:
                assert pe.value(pa[(i, j)]) == pe.value(pb[(i, j)]), (name, i, j)


def test_clustering_incremental_update_matches_fresh_build():
    _skip_if_no_glpk()
    df_indiv = pd.DataFrame({"c": ["c_1", "c_2", "c_3"], "t": [0, 0, 0], "cpu": [1.0, 2.0, 3.0]})
    instance = _Instance(df_indiv)

    # Prior-tick state, distinct from the target the incremental path lands on.
    u_prev = np.eye(3, dtype=int)
    w_prev = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    # Target state: asymmetric, non-degenerate weights.
    u_target = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int)
    w_target = np.array([[0, 1, 5], [1, 0, 2], [5, 2, 0]], dtype=float)

    fresh = PyomoModel({"solver": "glpk", "pb_number": 1}, instance)
    fresh.build(u_mat=u_target, w_mat=w_target)
    fresh.solve()
    fresh.fill_dual_values()

    incremental = PyomoModel({"solver": "glpk", "pb_number": 1}, instance)
    incremental.build(u_mat=u_prev, w_mat=w_prev)
    incremental.solve()  # simulates the previous tick's solve
    incremental.update_w(w_target)
    incremental.update_adjacency_constraints(u_target)
    incremental.solve()
    incremental.fill_dual_values()

    _assert_params_match(fresh, incremental, "w", "sol_u")
    assert pe.value(fresh.instance_model.obj) == pytest.approx(
        pe.value(incremental.instance_model.obj)
    )
    assert fresh.last_duals == incremental.last_duals


def test_placement_incremental_update_matches_fresh_build():
    _skip_if_no_glpk()
    df_indiv = pd.DataFrame(
        {
            "c": ["c_1", "c_2", "c_3"],
            "t": [0, 0, 0],
            "h": ["n_1", "n_1", "n_2"],
            "cpu": [1.0, 2.0, 3.0],
        }
    )
    df_meta = pd.DataFrame({"h": ["n_1", "n_2"], "cpu": [10.0, 10.0]})
    instance = _Instance(df_indiv, df_meta)

    u_prev = np.eye(3, dtype=int)
    v_prev = np.eye(3, dtype=int)
    dv_prev = np.zeros((3, 3))

    u_target = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=int)
    v_target = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int)
    dv_target = np.array([[0, 1, 5], [1, 0, 2], [5, 2, 0]], dtype=float)

    fresh = PyomoModel({"solver": "glpk", "pb_number": 2}, instance)
    fresh.build(u_mat=u_target, v_mat=v_target, dv_mat=dv_target)
    fresh.solve()
    fresh.fill_dual_values()

    incremental = PyomoModel({"solver": "glpk", "pb_number": 2}, instance)
    incremental.build(u_mat=u_prev, v_mat=v_prev, dv_mat=dv_prev)
    incremental.solve()  # simulates the previous tick's solve
    incremental.update_sol_u(u_target)
    incremental.update_obj_place(dv_target)
    incremental.update_adjacency_constraints(v_target)
    incremental.solve()
    incremental.fill_dual_values()

    _assert_params_match(fresh, incremental, "sol_u", "dv", "sol_v")
    assert pe.value(fresh.instance_model.obj) == pytest.approx(
        pe.value(incremental.instance_model.obj)
    )
    assert fresh.last_duals == incremental.last_duals
