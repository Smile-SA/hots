import pandas as pd
from typing import List

def build_df_from_containers(
    df_indiv: pd.DataFrame,
    tick_field: str,
    host_field: str,
    individual_field: str,
    metrics: List[str]
) -> pd.DataFrame:
    """
    Aggregate individual consumption into host-level time-series,
    ensuring each host has an entry for every tick (filling missing
    values with zeros). Flattens any MultiIndex columns correctly.
    """
    # 1) Pivot to wide
    df_pivot = (
        df_indiv
        .pivot_table(
            index=[tick_field, host_field],
            columns=individual_field,
            values=metrics,
            aggfunc='sum'
        )
        .fillna(0)
    )

    # 2) Reset index → all columns become MultiIndex (lvl2 empty for idx fields)
    df_pivot = df_pivot.reset_index()

    # 3) Flatten columns:
    #    - If tuple[1] is empty ('' or None), keep tuple[0]
    #    - Else combine as "{container}_{metric}"
    flat_cols = []
    for col in df_pivot.columns:
        if isinstance(col, tuple):
            metric, cid = col
            if cid in (None, ''):
                flat_cols.append(metric)
            else:
                flat_cols.append(f"{cid}_{metric}")
        else:
            flat_cols.append(col)
    df_pivot.columns = flat_cols

    # 4) Build full tick list
    all_ticks = pd.DataFrame({tick_field: df_pivot[tick_field].unique()})
    hosts = df_pivot[host_field].unique()

    # 5) For each host, merge full ticks (fills NaN → 0)
    rows = []
    for host in hosts:
        df_h = df_pivot[df_pivot[host_field] == host]
        df_full = all_ticks.merge(df_h, on=tick_field, how='left')
        df_full[host_field] = host
        rows.append(df_full.fillna(0))

    # 6) Concat and return
    return pd.concat(rows, ignore_index=True)
