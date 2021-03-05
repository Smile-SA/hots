"""
=========
cots tools
=========

Provide external tools usefull for the code.
"""

import pandas as pd


def change_max_dataset(df_indiv: pd.DataFrame,
                       container: str, col_indiv: str,
                       new_max: float, col_val: str,
                       col_time: str, min_time: int) -> pd.DataFrame:
    """Change dataset after max bound change."""
    working_df = df_indiv.loc[
        (df_indiv[col_time] >= min_time) & (
            df_indiv[col_indiv] == container)
    ]

    for row in working_df.iterrows():
        # print(row[1][col_val])
        if row[1][col_val] > new_max:
            df_indiv.loc[
                (df_indiv[col_time] == row[1][col_time]) & (
                    df_indiv[col_indiv] == container), col_val
            ] = new_max
