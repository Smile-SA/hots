"""HOTS visualization utilities."""

import matplotlib.pyplot as plt


def plot_clusters(df_host, labels, out_path):
    """Plot clustering results and save to the given path."""
    plt.figure()
    # TODO: implement scatter/time‑series plots
    plt.savefig(out_path)
    plt.close()
