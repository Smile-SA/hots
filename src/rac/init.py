import pandas as pd

# TODO list parameters here, or file to give in argument ?

# Global variables #

# We delete disk for the moment cause, not enough info
metrics = ['cpu', 'mem']

# Functions definitions #


def df_from_csv(filename: str) -> pd.DataFrame:
    """Load DataFrame from CSV file."""
    return pd.read_csv(
        filename, index_col=False)


# TODO check if files exist ?
# TODO optionnal node file ?
def init_dfs(data: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Perform CSV files reading in data folder."""
    return (df_from_csv(f'{data}/container_usage.csv'),
            df_from_csv(f'{data}/node_usage.csv'),
            df_from_csv(f'{data}/node_meta.csv'))
