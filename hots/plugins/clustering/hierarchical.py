# hots/plugins/clustering/hierarchical.py

"""Clustering plugin: agglomerative (hierarchical) clustering."""

from core.interfaces import ClusteringPlugin

import pandas as pd

from plugins.clustering.builder import build_matrix_indiv_attr

from scipy.cluster.hierarchy import fcluster, linkage


class HierarchicalClustering(ClusteringPlugin):
    """Hierarchical clustering plugin using SciPy linkage."""

    def __init__(self, parameters: dict, instance):
        """Initialize with linkage method and number of clusters."""
        self.method = parameters.get('method', 'ward')
        self.n_clusters = instance.config.clustering.nb_clusters
        self.tick_field = instance.config.tick_field
        self.indiv_field = instance.config.individual_field
        self.metrics = instance.config.metrics
        self.id_map = instance.get_id_map()

    def fit(self, df: pd.DataFrame) -> pd.Series:
        """Fit hierarchical clusters and return zero‐indexed labels."""
        mat = build_matrix_indiv_attr(
            df,
            self.tick_field,
            self.indiv_field,
            self.metrics,
            self.id_map,
        )
        z = linkage(mat.values, method=self.method)
        labels = fcluster(z, t=self.n_clusters, criterion='maxclust') - 1
        return pd.Series(labels, index=mat.index)
