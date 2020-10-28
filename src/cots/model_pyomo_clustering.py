"""
=========
cots small_pyomo
=========

One model instance, made as an example alternative to the
'model_small_cplex' example made with CPLEX.
"""

import pandas as pd

from pyomo import environ as pe

from . import init as it


class Model:
    """
    Class with a very small minimalist use of pyomo (for translation).

    Attributes :
    - mdl : pyomo Model (Abstract) Instance (basis)
    """

    def __init__(self, df_indiv: pd.DataFrame,
                 df_host_meta: pd.DataFrame):
        """Initialize Small_CPXInstance with data in Instance."""
        print('Building of small pyomo model ...')

        # Build the basis object "Model"
        self.mdl = pe.AbstractModel()

        # Prepare the sets and parameters
        self.build_parameters()

        # Build decision variables
        self.build_variables()

        # Build constraints of the problem
        self.build_constraints()

        # Build the objective function
        self.build_objective()

        # Put data in attribute
        self.create_data(df_indiv, df_host_meta)

        # Create the instance by feeding the model with the data
        self.instance_model = self.mdl.create_instance(self.data)
        self.instance_model.pprint()

    def build_parameters(self):
        """Build all Params and Sets."""
        # number of nodes
        self.mdl.n = pe.Param(within=pe.NonNegativeIntegers)
        # number of containers
        self.mdl.c = pe.Param(within=pe.NonNegativeIntegers)
        # number of data points
        self.mdl.t = pe.Param(within=pe.NonNegativeIntegers)
        # number of clusters
        self.mdl.k = pe.Param(within=pe.NonNegativeIntegers)

        # set of nodes
        self.mdl.N = pe.Set(dimen=1)
        # set of containers
        self.mdl.C = pe.Set(dimen=1)
        # set of clusters
        self.mdl.K = pe.Set(dimen=1)

        # capacity of nodes
        self.mdl.cap = pe.Param(self.mdl.N)
        # time window
        self.mdl.T = pe.Set(dimen=1)
        # containers usage
        self.mdl.cons = pe.Param(self.mdl.C, self.mdl.T)

    def build_variables(self):
        """Build all model variables."""
        # Variables Containers x Nodes
        self.mdl.x = pe.Var(self.mdl.C, self.mdl.N, domain=pe.Binary)
        # Variables Nodes
        self.mdl.a = pe.Var(self.mdl.N, domain=pe.Binary)
        # Delta
        self.mdl.delta = pe.Var(1, domain=pe.Binary)

        # Variables Containers x Clusters
        self.mdl.y = pe.Var(self.mdl.C, self.mdl.K, domain=pe.Binary)
        # Variables Containers x Containers
        self.mdl.u = pe.Var(self.mdl.C, self.mdl.C, domain=pe.Binary)
        # Variables Clusters
        self.mdl.b = pe.Var(self.mdl.K, domain=pe.Binary)

    def build_constraints(self):
        """Build all the constraints."""
        self.mdl.capacity = pe.Constraint(self.mdl.N, self.mdl.T,
                                          rule=capacity_)
        self.mdl.open_node = pe.Constraint(self.mdl.C, self.mdl.N,
                                           rule=open_node_)
        self.mdl.assignment = pe.Constraint(self.mdl.C,
                                            rule=assignment_)

    def build_objective(self):
        """Build the objective."""
        self.mdl.obj = pe.Objective(rule=open_nodes_)

    def create_data(self, df_indiv, df_host_meta):
        """Create data from dataframe."""
        self.cap = {}
        for n, n_data in df_host_meta.groupby([it.host_field]):
            self.cap.update({n: n_data['cpu'].values[0]})
        self.cons = {}
        df_indiv.reset_index(drop=True, inplace=True)
        for key, c_data in df_indiv.groupby([it.indiv_field, it.tick_field]):
            self.cons.update({key: c_data['cpu'].values[0]})

        self.data = {None: {
            'n': {None: df_indiv[it.host_field].nunique()},
            'c': {None: df_indiv[it.indiv_field].nunique()},
            't': {None: df_indiv[it.tick_field].nunique()},
            'N': {None: df_indiv[it.host_field].unique().tolist()},
            'C': {None: df_indiv[it.indiv_field].unique().tolist()},
            'cap': self.cap,
            'T': {None: df_indiv[it.tick_field].unique().tolist()},
            'cons': self.cons,
        }}

    def conso_n_t(self, mdl, node, t):
        """Express the total consumption of node at time t."""
        return sum(
            mdl.x[cont, node] * self.cons[cont][t]
            for cont in mdl.C)

    def mean(self, mdl, node):
        """Express the mean consumption of node."""
        return (sum(
            self.conso_n_t(node, t) for t in mdl.T
        ) / mdl.t)


def capacity_(mdl, node, time):
    """Express the capacity constraints."""
    return (sum(
        mdl.x[i, node] * mdl.cons[i, time] for i in mdl.C
    ) <= mdl.cap[node])


def open_node_(mdl, container, node):
    """Express the opening node constraint."""
    return mdl.x[container, node] <= mdl.a[node]


def assignment_(mdl, container):
    """Express the assignment constraint."""
    return sum(mdl.x[container, node] for node in mdl.N) == 1


def delta_1(mdl, node, time):
    """Express first inequality for delta."""
    # expr_n =
    # return ()


def delta_2(mdl, node, time):
    """Express second inequality for delta."""


def open_nodes_(mdl):
    """Express the numbers of open nodes."""
    return sum(mdl.a[i] for i in mdl.N)
