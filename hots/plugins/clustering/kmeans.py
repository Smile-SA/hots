# hots/plugins/clustering/kmeans.py

"""Clustering plugin: mini‐batch KMeans streaming."""

from core.interfaces import ClusteringPlugin

import pandas as pd

from plugins.clustering.builder import build_matrix_indiv_attr

from sklearn.cluster import MiniBatchKMeans


class StreamKMeans(ClusteringPlugin):
    """StreamKMeans plugin using scikit‐learn’s MiniBatchKMeans."""

    def __init__(self, parameters: dict, instance):
        """Initialize with number of clusters and instance config."""
        self.n_clusters = parameters['nb_clusters']
        self.tick_field = instance.config.tick_field
        self.indiv_field = instance.config.individual_field
        self.metrics = instance.config.metrics
        self.id_map = instance.get_id_map()
        self.model = MiniBatchKMeans(n_clusters=self.n_clusters)

    def fit(self, df: pd.DataFrame) -> pd.Series:
        """Partial‐fit on incoming data and return cluster labels."""
        mat = build_matrix_indiv_attr(
            df,
            self.tick_field,
            self.indiv_field,
            self.metrics,
            self.id_map,
        )
        labels = self.model.partial_fit(mat.values).predict(mat.values)
        return pd.Series(labels, index=mat.index)
