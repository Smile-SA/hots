from core.interfaces import ClusteringPlugin
from sklearn.cluster import SpectralClustering as SkSpectral
import pandas as pd
from plugins.clustering.builder import build_matrix_indiv_attr, build_similarity_matrix

class SpectralClustering(ClusteringPlugin):
    def __init__(self, parameters: dict, instance):
        self.n_clusters = parameters.get('nb_clusters', instance.config.clustering.nb_clusters)
        self.tick_field = instance.config.tick_field
        self.indiv_field = instance.config.individual_field
        self.metrics = instance.config.metrics
        self.id_map = instance.get_id_map()

    def fit(self, df: pd.DataFrame) -> pd.Series:
        mat = build_matrix_indiv_attr(
            df,
            self.tick_field,
            self.indiv_field,
            self.metrics,
            self.id_map
        )
        sim = build_similarity_matrix(mat)
        model = SkSpectral(n_clusters=self.n_clusters, affinity='precomputed')
        labels = model.fit_predict(sim)
        return pd.Series(labels, index=mat.index)
