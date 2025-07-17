from core.interfaces import ClusteringPlugin
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from plugins.clustering.builder import build_matrix_indiv_attr

class HierarchicalClustering(ClusteringPlugin):
    """
    Agglomerative (hierarchical) clustering plugin.
    Uses SciPy linkage + flat clustering into nb_clusters.
    """
    def __init__(self, parameters: dict, instance):
        # linkage method, e.g. 'ward', 'single', 'complete', etc.
        self.method = parameters.get('method', 'ward')
        # target #clusters from your AppConfig
        self.n_clusters = instance.config.clustering.nb_clusters

        # metadata from instance for matrix build
        self.tick_field = instance.config.tick_field
        self.indiv_field = instance.config.individual_field
        self.metrics = instance.config.metrics
        self.id_map = instance.get_id_map()

    def fit(self, df: pd.DataFrame) -> pd.Series:
        # build individual×tick matrix (rows sorted by id_map)
        mat = build_matrix_indiv_attr(
            df,
            self.tick_field,
            self.indiv_field,
            self.metrics,
            self.id_map
        )
        # compute linkage matrix
        Z = linkage(mat.values, method=self.method)
        # assign flat clusters (1..n_clusters) and zero‐index them
        labels = fcluster(Z, t=self.n_clusters, criterion='maxclust') - 1
        return pd.Series(labels, index=mat.index)
