"""
=========
cots model
=========

Define the optimization model we have, with its objective, constraints,
variables... It is built with pyomo, an open-source optimization modeling
language. We call the solver to solve our optimization problem from here.
"""

from pyomo.environ import AbstractModel


def create_model() -> AbstractModel:
    """Create the pyomo model to work with."""
    model = AbstractModel()
    return model
