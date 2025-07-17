import pandas as pd

def write_metrics(metrics: dict, path):
    df = pd.DataFrame([metrics])
    df.to_csv(path, mode='a', header=not Path(path).exists())
