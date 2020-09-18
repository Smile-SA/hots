"""
=========
cots small_pyomo
=========

One model instance, made as an exemple alternative to the
'model_small_cplex' exemple made with CPLEX.
"""

import pandas as pd

from pyomo import environ as pe


class Model:
    """
    Class with a very small minimalist use of pyomo (for translation).

    Attributes :
    - mdl : pyomo Model (Abstract) Instance (basis)
    """

    def __init__(self, df_containers: pd.DataFrame,
                 df_nodes_meta: pd.DataFrame):
        """Initialize Small_CPXInstance with data in Instance."""
        print('Building of small pyomo model ...')

        # Build the basis object "Model"
        self.mdl = pe.AbstractModel()

        # Prepare the Sets and parameters
        self.build_parameters()

        # Build decision variables
        self.build_variables()

        self.build_constraints()

        self.build_objective()

    def build_parameters(self):
        """Build all Params and Sets."""
        # number of nodes
        self.mdl.n = pe.Param(within=pe.NonNegativeIntegers)
        # number of containers
        self.mdl.c = pe.Param(within=pe.NonNegativeIntegers)
        # number of data points
        self.mdl.t = pe.Param(within=pe.NonNegativeIntegers)

        # set of nodes
        self.mdl.N = pe.RangeSet(1, self.mdl.n)
        # set of containers
        self.mdl.C = pe.RangeSet(1, self.mdl.c)

        # capacity of nodes
        self.mdl.cap = pe.Param(self.mdl.N)
        # time window
        self.mdl.T = pe.RangeSet(self.mdl.t)
        # containers usage
        self.mdl.cons = pe.Param(self.mdl.C, self.mdl.T)

    def build_variables(self):
        """Build all model variables."""
        # Variables Containers x Nodes
        self.mdl.x = pe.Var(self.mdl.C, self.mdl.N, domain=pe.Binary)

        # Variables Nodes
        self.mdl.a = pe.Var(self.mdl.N, domain=pe.Binary)

    def capacity_(self, node, time):
        """Express the capacity constraints."""
        return (sum(
            self.mdl.x[i, node] * self.mdl.cons[i, time] for i in self.mdl.C
        ) <= self.mdl.cap[node])

    def open_node_(self, container, node):
        """Express the opening node constraint."""
        return self.mdl.x[container, node] <= self.mdl.a[node]

    def assignment_(self, container):
        """Express the assignment constraint."""
        return sum(self.mdl.x[container, node] for node in self.mdl.N) == 1

    def build_constraints(self):
        """Build all the constraints."""
        self.mdl.capacity = pe.Constraint(self.mdl.N, self.mdl.T,
                                          rule=self.capacity_)
        self.mdl.open_node = pe.Constraint(self.mdl.C, self.mdl.C,
                                           rule=self.open_node_)
        self.mdl.assignment = pe.Constraint(self.mdl.C,
                                            rule=self.assignment_)

    def open_nodes_(self):
        """Expression for numbers of open nodes."""
        return sum(self.mdl.a[i] for i in self.mdl.N)

    def build_objective(self):
        """Build the objective."""
        self.mdl.obj = pe.Objective(rule=self.open_nodes_)
