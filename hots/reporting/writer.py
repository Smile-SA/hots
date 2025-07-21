import pandas as pd
from pathlib import Path

def write_metrics(metrics: dict, path):
    """
    Append a row of metrics to a CSV at 'path', creating parent
    directories if needed.
    """
    # 1) Ensure the directory exists
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # 2) Convert the metrics dict to a one‐row DataFrame
    df = pd.DataFrame([metrics])

    # 3) Append to the CSV, writing the header if the file is new
    df.to_csv(
        p,
        mode='a',
        header=not p.exists(),
        index=False
    )
