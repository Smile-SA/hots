import pandas as pd
from sklearn.metrics import silhouette_score

def eval_solutions(df_indiv, df_host, labels, clustering, optimization, heuristic, instance):
    # silhouette
    sil = silhouette_score(df_host.drop([instance.config.tick_field, instance.config.host_field], axis=1), labels)
    # detect conflicts via similarity matrix, capacity, etc.
    # return (solution2, {'silhouette': sil, ...})
    return solution2, {'silhouette': sil}
