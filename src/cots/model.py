"""
=========
cots model
=========

Define the optimization model we have, with its objective, constraints,
variables... It is built with pyomo, an open-source optimization modeling
language. We call the solver to solve our optimization problem from here.
"""

import importlib.util
from abc import ABCMeta, abstractmethod
from pathlib import Path

from .instance import Instance


class ModelInterface(metaclass=ABCMeta):
    """
    Interface class for defining the optimization problem.

    It must at least have the following attributes :
    - mdl :

    and the following implemented methods :
    - build_variables :
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        """Define requirements for being a good instance of Model."""
        return (hasattr(subclass, 'build_variables')
                and callable(subclass.build_variables))

    @abstractmethod
    def __init__(self):
        """Initialize the model."""
        raise NotImplementedError

    @abstractmethod
    def build_variables(self):
        """Build all model variables."""
        raise NotImplementedError


def load_model_file(model_file: str):
    """Load the file in which we find the model."""
    module_name = Path(model_file).stem
    spec = importlib.util.spec_from_file_location(
        module_name, model_file)
    spec_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spec_model)
    return spec_model


def create_model(model_file: str, my_instance: Instance):
    """Create the pyomo model to work with."""
    spec_model = load_model_file(model_file)
    model = spec_model.Model(my_instance.df_containers,
                             my_instance.df_nodes_meta)
    print(issubclass(spec_model.Model, ModelInterface))
    print(model.mdl.n)
    print(model.mdl.N)
    return model
