"""Pyomo‑based optimization plugin for HOTS."""

from hots.core.interfaces import OptimizationPlugin

from .model import Model as PyomoModelClass


class PyomoModel(OptimizationPlugin):
    """Optimization plugin using Pyomo."""

    def __init__(self, params: dict, instance):
        """Initialize with parameters and HOTS instance."""
        self.instance = instance
        self.pb_number = params.get('pb_number', 1)
        self.solver = params.get('solver', 'glpk')
        self.verbose = params.get('verbose', False)

    def solve(self, df_host, labels):
        """Build, solve, and return a HOTS Model."""
        model = PyomoModelClass(
            self.pb_number,
            self.instance.df_indiv,
            self.instance.config.metrics[0],
            self.instance.get_id_map(),
            self.instance.df_meta,
            nb_clusters=self.instance.config.clustering.nb_clusters,
            verbose=self.verbose,
        )
        model.solve(solver=self.solver)
        model.labels = labels
        return model
