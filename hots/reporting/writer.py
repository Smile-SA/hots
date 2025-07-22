"""HOTS reporting writer."""

from pathlib import Path

import pandas as pd


def write_metrics(metrics: dict, path):
    """Append a row of metrics to a CSV at path.

    Creates parent directories if needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([metrics])
    df.to_csv(
        p,
        mode='a',
        header=not p.exists(),
        index=False
    )
