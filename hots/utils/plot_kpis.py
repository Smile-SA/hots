from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_metrics(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file: {path}")


def _plot_series(df: pd.DataFrame, x: str, y: str, outdir: Path) -> None:
    if y not in df.columns or x not in df.columns:
        return
    s = df[[x, y]].dropna()
    if s.empty:
        return

    fig, ax = plt.subplots()
    ax.plot(s[x], s[y])
    ax.set_title(y)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{y}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_file", type=Path, help="metrics.csv or metrics.parquet")
    ap.add_argument("--x", default="tick_end", help="x axis column (default tick_end)")
    ap.add_argument("--outdir", type=Path, default=Path("kpi_plots"))
    args = ap.parse_args()

    df = _read_metrics(args.metrics_file)

    # Choose a default set of important KPIs to plot
    candidates = [
        "plc__moved_ratio",
        "moves__moves_count",
        "clst_struct__n_clusters",
        "clst_struct__size_gini",
        "place_conf__density",
        "place_conf__deg_p95",
        "load__host_load_cv",
        "load__host_load_gini",
    ]

    # fallback x if missing
    x = args.x if args.x in df.columns else ("tick" if "tick" in df.columns else df.columns[0])

    for y in candidates:
        _plot_series(df, x=x, y=y, outdir=args.outdir)

    # Also save a quick summary table
    summary = df.describe(include="all")
    summary.to_csv(args.outdir / "summary.csv")


if __name__ == "__main__":
    main()
