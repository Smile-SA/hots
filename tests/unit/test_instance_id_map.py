"""Unit tests for the append-only container ID map."""

import pandas as pd

from hots.core.instance import FieldNames, Instance


class _ConnectorCfg:
    parameters = {}


class _Cfg:
    kafka = None
    connector = _ConnectorCfg()


def _instance(df):
    inst = Instance.__new__(Instance)
    inst.config = _Cfg()
    inst.fields = FieldNames(
        tick="timestamp", host="machine_id", individual="container_id", metrics=("cpu",)
    )
    inst.df_indiv = df
    inst._id_map = {}
    inst._inv_id_map = {}
    inst._id_map_stale = True
    return inst


def _rows(pairs):
    return pd.DataFrame(
        [{"timestamp": t, "container_id": c, "machine_id": "n_1", "cpu": 1.0} for t, c in pairs]
    )


def test_indices_are_stable_when_a_new_id_sorts_first():
    """A container arriving with a lexically smaller ID must not shift existing indices."""
    inst = _instance(_rows([(0, "c_b"), (0, "c_c")]))
    first = dict(inst.get_id_map())
    assert first == {"c_b": 0, "c_c": 1}

    # 'c_a' sorts before every existing ID
    inst.df_indiv = pd.concat([inst.df_indiv, _rows([(1, "c_a")])], ignore_index=True)
    inst._invalidate_caches()
    second = inst.get_id_map()

    assert second["c_b"] == first["c_b"]
    assert second["c_c"] == first["c_c"]
    assert second["c_a"] == 2  # new IDs take the next free index
    assert inst.get_inv_id_map() == {0: "c_b", 1: "c_c", 2: "c_a"}


def test_id_map_is_deterministic_within_a_batch():
    """Unseen IDs of the same batch are assigned in sorted order."""
    inst = _instance(_rows([(0, "c_z"), (0, "c_a"), (0, "c_m")]))
    assert inst.get_id_map() == {"c_a": 0, "c_m": 1, "c_z": 2}


def test_repeated_calls_do_not_reassign():
    """get_id_map() is idempotent when no new data arrived."""
    inst = _instance(_rows([(0, "c_1"), (0, "c_2")]))
    assert inst.get_id_map() == inst.get_id_map()
    inst._invalidate_caches()
    assert inst.get_id_map() == {"c_1": 0, "c_2": 1}
