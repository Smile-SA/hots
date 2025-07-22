# hots/plugins/clustering/custom_spectral.py

"""Clustering plugin: custom spectral clustering for HOTS."""

from hots.core.interfaces import ClusteringPlugin
from hots.plugins.clustering.builder import (
    build_matrix_indiv_attr,
    build_similarity_matrix,
)

import numpy as np
from numpy.linalg import multi_dot

import pandas as pd

from scipy.linalg import fractional_matrix_power
from scipy.linalg.lapack import dsyevr


class CustomSpectralClustering(ClusteringPlugin):
    """Custom spectral clustering plugin using normalized Laplacian."""

    def __init__(self, parameters: dict, instance):
        """Initialize with cluster count and instance configuration."""
        self.n_clusters = parameters.get(
            'nb_clusters',
            instance.config.clustering.nb_clusters,
        )
        self.tick_field = instance.config.tick_field
        self.indiv_field = instance.config.individual_field
        self.metrics = instance.config.metrics
        self.id_map = instance.get_id_map()

    def fit(self, df: pd.DataFrame) -> pd.Series:
        """Compute labels by eigen-decomposing the normalized Laplacian."""
        x = build_matrix_indiv_attr(
            df,
            self.tick_field,
            self.indiv_field,
            self.metrics,
            self.id_map,
        )
        w = build_similarity_matrix(x)
        d = np.diag(w.sum(axis=1))
        d_inv_sqrt = fractional_matrix_power(d, -0.5)
        var_l = multi_dot([d_inv_sqrt, w, d_inv_sqrt])
        eigvals, eigvecs, _ = dsyevr(var_l, range='A')
        idx = np.argsort(eigvals)[::-1][:self.n_clusters]
        u = eigvecs[:, idx]
        labels = (np.arange(len(u)) % self.n_clusters).astype(int)
        return pd.Series(labels, index=x.index)
