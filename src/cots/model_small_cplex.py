"""
=========
cots model
=========

Define the optimization model we have, with its objective, constraints,
variables, and build it from the ``Instance``. Provide all optim model
related methods.
The optimization model description is based on docplex, the Python API of
CPLEX, own by IBM.
"""

# from itertools import combinations
# from typing import Dict

from docplex.mp.model import Model

# import numpy as np

import pandas as pd

# from .init import metrics
# from .instance import Instance


class SmallCPXInstance:
    """
    Class with a very small minimalist use of CPLEX (for translation).

    Attributes :
    - mdl : DOCPLEX Model Instance (basis)
    - nodes_names : names of nodes related variables
    - containers_names : names of containers related variables
    """

    def __init__(self, df_containers: pd.DataFrame,
                 df_nodes_meta: pd.DataFrame):
        """Initialize Small_CPXInstance with data in Instance."""
        print('Building of small CPLEX model ...')
        self.time_window = df_containers['timestamp'].nunique()

        # Build the basis object "Model"
        self.mdl = Model(name='small_allocation', cts_by_name=False)

        # Prepare the variables names depending on data we have
        self.build_names(df_nodes_meta, df_containers)

        # Build CPLEX variables with their names
        self.build_variables()

        # Build data as dict for easy use by CPLEX
        self.build_data(df_containers, df_nodes_meta)

        # Build the constraints describing our small allocation problem
        self.build_constraints()

        # Build the objective aimed by our problem
        self.build_objective()

        # We export the model to a LP file (standard in Optimization)
        self.mdl.export_as_lp(path='./small_allocation.lp')

    def build_names(self, df_nodes_meta, df_containers):
        """Build only names variables from nodes and containers."""
        self.nodes_names = df_nodes_meta['machine_id'].unique()
        self.containers_names = df_containers['container_id'].unique()

    def build_variables(self):
        """Build all model variables."""
        ida = list(self.nodes_names)
        self.mdl.a = self.mdl.binary_var_dict(ida, name=lambda k: 'a_%s' % k)

        idx = [(c, n) for c in self.containers_names for n in self.nodes_names]
        self.mdl.x = self.mdl.binary_var_dict(
            idx, name=lambda k: 'x_%s,%s' % (k[0], k[1]))

    def build_data(self, df_containers: pd.DataFrame,
                   df_nodes_meta: pd.DataFrame):
        """Build all model data from instance."""
        self.containers_data = {}
        self.nodes_data = {}
        for n in self.nodes_names:
            self.nodes_data[n] = df_nodes_meta.loc[
                df_nodes_meta['machine_id'] == n,
                ['cpu']].to_numpy()[0]
        for c in self.containers_names:
            self.containers_data[c] = df_containers.loc[
                df_containers['container_id'] == c,
                ['timestamp', 'cpu']].to_numpy()

    def build_constraints(self):
        """Build all model constraints."""
        for node in self.nodes_names:
            for t in range(self.time_window):
                i = 0
                # Capacity constraint
                self.mdl.add_constraint(self.mdl.sum(
                    self.mdl.x[c, node] * self.containers_data[c][t][i]
                    for c in self.
                    containers_names) <= self.nodes_data[node][i - 1],
                    'capacity_' + str(node) + '_' + str(t))

            # Assign constraint (x[c,n] = 1 => a[n] = 1)
            for c in self.containers_names:
                self.mdl.add_constraint(
                    self.mdl.x[c, node] <= self.mdl.a[node],
                    'open_a_' + str(node))

        # Container assignment constraint (1 and only 1 x[c,_] = 1 for all c)
        for container in self.containers_names:
            self.mdl.add_constraint(self.mdl.sum(
                self.mdl.x[container, node] for node in self.nodes_names) == 1,
                'assign_' + str(container))

    def build_objective(self):
        """Build objective."""
        # Minimize sum of a[n] (number of open nodes)
        self.mdl.minimize(self.mdl.sum(
            self.mdl.a[n] for n in self.nodes_names))

    def solve(self):
        """Solve the given problem."""
        if not self.mdl.solve(log_output=True):
            print('*** Problem has no solution ***')
        else:
            print('*** Model solved as function:')
            self.mdl.print_solution()
            self.mdl.report()
            print(self.mdl.objective_value)
