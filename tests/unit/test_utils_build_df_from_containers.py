"""Unit tests for hots.utils.tools.build_df_from_containers."""

from hots.utils.tools import build_df_from_containers

import pandas as pd


def test_build_df_from_containers_fills_hosts_and_metrics():
    """
    Basic sanity checks for the host×tick aggregation:

    - Output has tick, host and metric columns (no 'container' column).
    - No NaNs in host/metric columns.
    - Ticks are sorted.
    - Aggregated metric values are consistent with the input.
    """
    df_indiv = pd.DataFrame(
        [
            {'tick': 1, 'container': 'c1', 'host': 'h1', 'cpu': 1.0},
            {'tick': 3, 'container': 'c1', 'host': 'h1', 'cpu': 3.0},
            {'tick': 2, 'container': 'c2', 'host': 'h2', 'cpu': 2.0},
        ]
    )

    out = build_df_from_containers(
        df_indiv=df_indiv,
        tick_field='tick',
        host_field='host',
        metrics=['cpu'],
    )

    # 1) Expected columns: tick, host and the metric(s)
    assert set(out.columns) == {'tick', 'host', 'cpu'}

    # 2) No missing host/metric entries after filling
    assert out['host'].notna().all()
    assert out['cpu'].notna().all()

    # 3) Ticks are globally sorted
    ticks = out['tick'].tolist()
    assert ticks == sorted(ticks)

    # 4) Check basic aggregation: sum of cpu across hosts per tick
    #    should match the sum of input cpu per tick.
    input_sum = df_indiv.groupby('tick')['cpu'].sum().sort_index()
    out_sum = out.groupby('tick')['cpu'].sum().sort_index()
    # Compare index and values
    assert list(input_sum.index) == list(out_sum.index)
    assert all(abs(a - b) < 1e-9 for a, b in zip(input_sum.values, out_sum.values))
