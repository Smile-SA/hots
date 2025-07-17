# hots/plugins/clustering/custom_spectral.py

from core.interfaces import ClusteringPlugin
from plugins.clustering.builder import build_matrix_indiv_attr, build_similarity_matrix
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.linalg.lapack import dsyevr
from numpy.linalg import multi_dot
import pandas as pd

class CustomSpectralClustering(ClusteringPlugin):
    """
    Custom spectral clustering: compute normalized Laplacian,
    extract top‐k eigenvectors, simple label assignment.
    """
    def __init__(self, parameters: dict, instance):
        self.n_clusters = parameters.get(
            'nb_clusters',
            instance.config.clustering.nb_clusters
        )
        self.tick_field = instance.config.tick_field
        self.indiv_field = instance.config.individual_field
        self.metrics = instance.config.metrics
        self.id_map = instance.get_id_map()

    def fit(self, df: pd.DataFrame) -> pd.Series:
        # 1) build individual×tick matrix
        X = build_matrix_indiv_attr(
            df,
            self.tick_field,
            self.indiv_field,
            self.metrics,
            self.id_map
        )
        # 2) compute similarity and degree matrix
        W = build_similarity_matrix(X)
        D = np.diag(W.sum(axis=1))

        # 3) normalized Laplacian: D^(-1/2) @ W @ D^(-1/2)
        D_inv_sqrt = fractional_matrix_power(D, -0.5)
        L = multi_dot([D_inv_sqrt, W, D_inv_sqrt])

        # 4) eigen-decomposition
        eigvals, eigvecs, _ = dsyevr(L, range='A')
        # pick the top k eigenvectors
        idx = np.argsort(eigvals)[::-1][: self.n_clusters]
        U = eigvecs[:, idx]

        # 5) simple label assignment (you can swap in k-means on U here)
        labels = (np.arange(len(U)) % self.n_clusters).astype(int)

        return pd.Series(labels, index=X.index)
