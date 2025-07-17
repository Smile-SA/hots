from core.interfaces import OptimizationPlugin
from .model import Model as PyomoModelClass

class PyomoModel(OptimizationPlugin):
    def __init__(self, params, instance):
        self.instance = instance
        self.pb_number = params.get('pb_number',1)
        self.solver = params.get('solver','glpk')
        self.verbose = params.get('verbose',False)

    def solve(self, df_host, labels):
        model = PyomoModelClass(
            self.pb_number,
            self.instance.df_indiv,
            self.instance.config.metrics[0],
            self.instance.get_id_map(),
            self.instance.df_meta,
            nb_clusters=self.instance.config.clustering.nb_clusters,
            verbose=self.verbose
        )
        model.solve(solver=self.solver)
        model.labels = labels
        return model
