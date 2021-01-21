"""
=========
cots allocation
=========

Provide resource allocation related functions to handle this problem.
"""

from typing import Dict

import pandas as pd

from . import init as it


def check_constraints(config: Dict, df_host: pd.DataFrame) -> bool:
    """Check if allocation constraints are satisfied or not."""
    satisfied = True
    print(config)
    print(df_host)
    if df_host[it.host_field].nunique() > config['objective']['open_nodes']:
        print('Too many open nodes !')
        satisfied = False
    elif df_host[it.host_field].nunique() < config['objective']['open_nodes']:
        print('Less open nodes than the objective !')
        satisfied = False
    else:
        print('Right number of open nodes.')

    if max(df_host[it.metrics[0]]) > config['objective']['target_load_CPU']:
        print('Max resources used > wanted max !')
        satisfied = False
    else:
        print('Max used resources ok.')

    return satisfied
