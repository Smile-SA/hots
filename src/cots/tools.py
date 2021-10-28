"""
=========
cots tools
=========

Provide external tools usefull for the code.
"""

import sys
from typing import List

import pandas as pd


def print_size_vars(list_vars: List):
    """Display size (memory) of variables."""
    print('List of variables with sizes:')
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list_vars),
                             key=lambda x: -x[1])[:10]:
        print('{:>30}: {:>8}'.format(name, sizeof_fmt(size)))


def sizeof_fmt(num, suffix='B'):
    """Ease display of memory sizes."""
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return '%3.1f %s%s' % (num, unit, suffix)
        num /= 1024.0
    return '%.1f %s%s' % (num, 'Yi', suffix)


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
