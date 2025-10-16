# hots/plugins/clustering/kmeans.py

"""Clustering plugin: mini‐batch KMeans streaming."""

from typing import Any, Dict

from hots.core.interfaces import ClusteringPlugin
from hots.plugins.clustering.builder import build_matrix_indiv_attr

import pandas as pd

from sklearn.cluster import MiniBatchKMeans


class StreamKMeans(ClusteringPlugin):
    """StreamKMeans plugin using scikit‐learn’s MiniBatchKMeans."""

    def __init__(self, params: Dict[str, Any], instance):
        """Initialize the streaming k-means plugin."""
        self.n_clusters = params.get('nb_clusters', 5)
        self.batch_size = params.get('batch_size', 100)
        self.random_state = params.get('random_state', None)
        self.tick_field = instance.config.tick_field
        self.indiv_field = instance.config.individual_field
        self.metrics = instance.config.metrics
        self.id_map = instance.get_id_map()
        self.model = None

    def fit(self, df: pd.DataFrame) -> pd.Series:
        """
        Rebuild and fit a MiniBatchKMeans on the current data, then return labels.
        This avoids any mismatch in expected feature dimension.
        """
        mat = build_matrix_indiv_attr(
            df,
            self.tick_field,
            self.indiv_field,
            self.metrics,
            self.id_map,
        )
        x = mat.values
        # rebuild the model now that X.shape[1] is known
        self.model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        labels = self.model.fit_predict(x)
        return pd.Series(labels, index=mat.index)
