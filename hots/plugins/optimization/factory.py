# hots/plugins/optimization/factory.py

"""HOTS Factory for building optimization models."""


class OptimizationFactory:
    """Factory for optimization backends (Pyomo, OR-Tools, etc.)."""

    @staticmethod
    def create(cfg, instance, *, pb_number: int = 1, backend: str | None = None):
        """
        Create and return an optimization plugin for the specified problem number.
        pb_number: 1 for clustering, 2 for business problem.
        backend: optional override ('pyomo', later maybe 'ortools', 'pulp', etc.).
        """
        # Determine backend: explicit param > cfg.backend > default
        be = (backend or getattr(cfg, 'backend', None) or 'pyomo').lower()

        if be == 'pyomo':
            from .pyomo_model import PyomoModel
            params = dict(cfg.parameters)
            params['pb_number'] = pb_number
            params['solver'] = getattr(cfg, 'solver', 'glpk')
            return PyomoModel(params=params, instance=instance)

        raise ValueError(f'Unknown optimization backend: {be}')
