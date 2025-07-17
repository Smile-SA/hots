import matplotlib.pyplot as plt

def plot_clusters(df_host, labels, out_path):
    plt.figure()
    # your scatter/time-series plots
    plt.savefig(out_path)
    plt.close()
