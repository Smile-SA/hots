"""Unit tests for bounded, incremental streaming updates on Instance."""

import pandas as pd

from hots.core.instance import FieldNames, Instance
from hots.utils.tools import build_df_from_containers

TICK, INDIV, HOST, METRIC = "timestamp", "container_id", "machine_id", "cpu"


class _ConnectorCfg:
    def __init__(self, **overrides):
        # Only loop/connector concerns live here; field names come from `data`.
        self.parameters = {
            "window_duration": 3,
            "tick_increment": 1,
        }
        self.parameters.update(overrides)


class _Cfg:
    def __init__(self, **overrides):
        self.kafka = None
        self.connector = _ConnectorCfg(**overrides)


def _rows(specs):
    """specs: list of (tick, container, host, cpu)."""
    return pd.DataFrame([{TICK: t, INDIV: c, HOST: h, METRIC: v} for t, c, h, v in specs])


def _instance(df, **cfg_overrides):
    inst = Instance.__new__(Instance)
    inst.config = _Cfg(**cfg_overrides)
    inst.fields = FieldNames(tick=TICK, host=HOST, individual=INDIV, metrics=(METRIC,))
    inst._id_map = {}
    inst._inv_id_map = {}
    inst._id_map_stale = True
    inst._known_hosts = set(df[HOST].unique())
    inst.df_indiv = df
    inst.df_host = build_df_from_containers(df, TICK, HOST, [METRIC])
    return inst


def _full_rebuild(df):
    return build_df_from_containers(df, TICK, HOST, [METRIC])


def test_incremental_host_aggregate_matches_full_rebuild():
    """The incremental df_host must equal a rebuild over the retained history."""
    inst = _instance(
        _rows([(0, "c_1", "n_1", 1.0), (0, "c_2", "n_2", 2.0)]),
        history_ticks=100,  # no pruning, so we can compare against a full rebuild
    )
    inst.update_data(_rows([(1, "c_1", "n_1", 3.0), (1, "c_2", "n_2", 4.0)]))
    inst.update_data(_rows([(2, "c_1", "n_2", 5.0), (2, "c_2", "n_2", 6.0)]))

    expected = _full_rebuild(inst.df_indiv)
    got = inst.df_host.sort_values([TICK, HOST], ignore_index=True)
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_split_tick_batches_are_re_aggregated():
    """A tick arriving across two batches must be summed, not double-rowed."""
    inst = _instance(_rows([(0, "c_1", "n_1", 1.0)]), history_ticks=100)

    # tick 1 arrives in two pieces, as Kafka message batches can
    inst.update_data(_rows([(1, "c_1", "n_1", 2.0)]))
    inst.update_data(_rows([(1, "c_2", "n_1", 5.0)]))

    row = inst.df_host[(inst.df_host[TICK] == 1) & (inst.df_host[HOST] == "n_1")]
    assert len(row) == 1, "one row per (tick, host)"
    assert row[METRIC].iloc[0] == 7.0

    pd.testing.assert_frame_equal(
        inst.df_host.sort_values([TICK, HOST], ignore_index=True),
        _full_rebuild(inst.df_indiv),
        check_dtype=False,
    )


def test_idle_host_still_gets_zero_rows():
    """A host with no containers in the new slice keeps a zero-filled row."""
    inst = _instance(
        _rows([(0, "c_1", "n_1", 1.0), (0, "c_2", "n_2", 2.0)]),
        history_ticks=100,
    )
    # at tick 1 both containers sit on n_1, so n_2 is idle
    inst.update_data(_rows([(1, "c_1", "n_1", 1.0), (1, "c_2", "n_1", 2.0)]))

    idle = inst.df_host[(inst.df_host[TICK] == 1) & (inst.df_host[HOST] == "n_2")]
    assert len(idle) == 1
    assert idle[METRIC].iloc[0] == 0.0


def test_history_is_bounded():
    """df_indiv/df_host must not grow without bound over a long run."""
    inst = _instance(_rows([(0, "c_1", "n_1", 1.0)]))
    retention = inst.history_ticks
    assert retention == 3 + 1 + 1  # window_duration + tick_increment + slack

    for t in range(1, 50):
        inst.update_data(_rows([(t, "c_1", "n_1", 1.0)]))

    assert inst.df_indiv[TICK].min() == 49 - retention + 1
    assert len(inst.df_indiv) == retention
    assert len(inst.df_host) == retention
    # every tick the working window can ask for is still there
    assert inst.get_working_df(49 - 2, 49)[TICK].nunique() == 3


def test_no_window_duration_keeps_full_history():
    inst = _instance(_rows([(0, "c_1", "n_1", 1.0)]), window_duration=None)
    assert inst.history_ticks is None
    for t in range(1, 10):
        inst.update_data(_rows([(t, "c_1", "n_1", 1.0)]))
    assert len(inst.df_indiv) == 10


def test_empty_batch_is_a_noop():
    inst = _instance(_rows([(0, "c_1", "n_1", 1.0)]))
    before = inst.df_indiv.copy()
    inst.update_data(pd.DataFrame())
    pd.testing.assert_frame_equal(inst.df_indiv, before)
