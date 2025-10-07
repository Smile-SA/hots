"""HOTS preprocessing tools."""

from typing import List

import pandas as pd


def build_df_from_containers(
    df_indiv: pd.DataFrame,
    tick_field: str,
    host_field: str,
    metrics: List[str]
) -> pd.DataFrame:
    """Aggregate individual consumption into host-level time-series.
    Ensures each host has an entry for every tick (filling missing values
    with zeros).
    """
    # Group by timestamp and host, then sum the metrics
    df_agg = (
        df_indiv
        .groupby([tick_field, host_field])[metrics]
        .sum()
        .reset_index()
    )

    # Ensure all timestamp-host combinations exist (fill missing with 0)
    unique_ticks = df_indiv[tick_field].unique()
    unique_hosts = df_indiv[host_field].unique()

    # Create all combinations
    all_combinations = pd.MultiIndex.from_product(
        [unique_ticks, unique_hosts],
        names=[tick_field, host_field]
    ).to_frame(index=False)

    # Merge and fill missing values with 0
    df_result = (
        all_combinations
        .merge(df_agg, on=[tick_field, host_field], how='left')
        .fillna(0)
        .sort_values([tick_field, host_field])
        .reset_index(drop=True)
    )

    return df_result
