import pandas as pd
from core.interfaces import ClusteringPlugin
from sklearn.cluster import MiniBatchKMeans
from plugins.clustering.builder import build_matrix_indiv_attr

class StreamKMeans(ClusteringPlugin):
    def __init__(self, parameters, instance):
        self.n_clusters = parameters['nb_clusters']
        self.tick_field = instance.config.tick_field
        self.indiv_field = instance.config.individual_field
        self.metrics = instance.config.metrics
        self.id_map = instance.get_id_map()
        self.model = MiniBatchKMeans(n_clusters=self.n_clusters)

    def fit(self, df):
        mat = build_matrix_indiv_attr(df, self.tick_field, self.indiv_field, self.metrics, self.id_map)
        labels = self.model.partial_fit(mat.values).predict(mat.values)
        return pd.Series(labels, index=mat.index)
