"""HOTS preprocessing tools."""

import logging
from typing import List

import numpy as np

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


def slice_by_time(df, col, tmin, tmax):
    """Get data between tmin and tmax in df."""
    return df[(df[col] >= tmin) & (df[col] <= tmax)]


def check_missing_entries_df(
    df: pd.DataFrame,
    tick_field: str,
    indiv_field: str,
    host_field: str,
    metrics: list[str],
) -> pd.DataFrame:
    """Ensure all (timestamp, container) pairs exist and fill missing data."""
    # 1. Prepare unique sorted values
    all_timestamps = np.sort(df[tick_field].unique())
    all_containers = np.sort(df[indiv_field].unique())

    # 2. Build full index and reindex
    full_index = pd.MultiIndex.from_product(
        [all_timestamps, all_containers],
        names=[tick_field, indiv_field],
    )

    df = df.set_index([tick_field, indiv_field]).reindex(full_index).reset_index()

    # 3. Forward/backward fill host (intra-container)
    df[host_field] = (
        df.groupby(indiv_field)[host_field]
          .ffill()
          .bfill()
    )

    # 4. Fill metrics with zeros
    df[metrics] = df[metrics].fillna(0.0)

    # 5. Sort rows chronologically
    df.sort_values(by=[tick_field, indiv_field], ignore_index=True, inplace=True)

    return df
