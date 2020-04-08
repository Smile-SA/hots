"""
=========
rac init
=========

Provide stuff for initialization step (load DataFrames,
global variables)
"""

import typing as t
from pathlib import Path

import pandas as pd

# TODO list parameters here, or file to give in argument ?

# Global variables #

# We delete disk for the moment cause, not enough info
metrics = ['cpu', 'mem']

# Functions definitions #


def df_from_csv(file: Path) -> pd.DataFrame:
    """Load DataFrame from CSV file."""
    return pd.read_csv(
        file, index_col=False)


# TODO check if files exist ?
# TODO optionnal node file ?
def init_dfs(data: str) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform CSV files reading in data folder."""
    p_data = Path(data)
    return (df_from_csv(p_data / 'container_usage.csv'),
            df_from_csv(p_data / 'node_usage.csv'),
            df_from_csv(p_data / 'node_meta.csv'))
