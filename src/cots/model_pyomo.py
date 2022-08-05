"""
=========
cots small_pyomo
=========

One model instance, made as an example alternative to the
'model_small_cplex' example made with CPLEX.
"""

from typing import Dict

import pandas as pd

from pyomo import environ as pe

import numpy as np

from . import init as it


class Model:
    """
    Class with a very small minimalist use of pyomo (for translation).

    Attributes :
    - mdl : pyomo Model (Abstract) Instance (basis)
    """

    def __init__(self, pb_number: int,
                 df_indiv: pd.DataFrame,
                 dict_id_c: Dict,
                 df_host_meta: pd.DataFrame = None,
                 nb_clusters: int = None,
                 w: np.array = None):
        """Initialize Pyomo model with data in Instance."""
        print('Building of pyomo model ...')

        # Which problem we build :
        # - 1 = only clustering
        # - 2 = placement from clustering
        self.pb_number = pb_number

        # Build the basis object "Model"
        self.mdl = pe.AbstractModel()

        # Prepare the sets and parameters
        w_d = dict(
            ((j, i), w[i][j]) for i in range(len(w)) for j in range(len(w[0]))
        )
        self.build_parameters(w_d)

        # Build decision variables
        self.build_variables()

        # Build constraints of the problem
        self.build_constraints()

        # Build the objective function
        self.build_objective()

        # Put data in attribute
        self.create_data(df_indiv, dict_id_c,
                         df_host_meta,
                         nb_clusters
                         )

        # Create the instance by feeding the model with the data
        self.instance_model = self.mdl.create_instance(self.data)
        # self.instance_model.pprint()
        # self.instance_model.write('place_pyomo.lp')

    def build_parameters(self, w):
        """Build all Params and Sets."""
        # number of containers
        self.mdl.c = pe.Param(within=pe.NonNegativeIntegers)
        # set of containers
        self.mdl.C = pe.Set(dimen=1)

        # clustering case
        if self.pb_number == 1:
            # number of clusters
            self.mdl.k = pe.Param(within=pe.NonNegativeIntegers)
            # set of clusters
            self.mdl.K = pe.Set(dimen=1)
            # distances
            self.mdl.w = pe.Param(self.mdl.C, self.mdl.C,
                                  initialize=w)
        # placement case
        elif self.pb_number == 2:
            # number of nodes
            self.mdl.n = pe.Param(within=pe.NonNegativeIntegers)
            # number of data points
            self.mdl.t = pe.Param(within=pe.NonNegativeIntegers)
            # set of nodes
            self.mdl.N = pe.Set(dimen=1)
            # capacity of nodes
            self.mdl.cap = pe.Param(self.mdl.N)
            # time window
            self.mdl.T = pe.Set(dimen=1)
            # containers usage
            self.mdl.cons = pe.Param(self.mdl.C, self.mdl.T)

    def build_variables(self):
        """Build all model variables."""
        if self.pb_number == 1:
            # Variables Containers x Clusters
            self.mdl.y = pe.Var(self.mdl.C, self.mdl.K,
                                domain=pe.NonNegativeReals,
                                bounds=(0, 1))
            # Variables Containers x Containers
            self.mdl.u = pe.Var(self.mdl.C, self.mdl.C,
                                domain=pe.NonNegativeReals,
                                bounds=(0, 1))
            # Variables Clusters
            self.mdl.b = pe.Var(self.mdl.K,
                                domain=pe.NonNegativeReals,
                                bounds=(0, 1))
        elif self.pb_number == 2:
            # Variables Containers x Nodes
            self.mdl.x = pe.Var(self.mdl.C, self.mdl.N,
                                domain=pe.NonNegativeReals,
                                bounds=(0, 1))
            # Variables Nodes
            self.mdl.a = pe.Var(self.mdl.N,
                                domain=pe.NonNegativeReals,
                                bounds=(0, 1))
            # Variables Containers x Containers
            self.mdl.v = pe.Var(self.mdl.C, self.mdl.C,
                                domain=pe.NonNegativeReals,
                                bounds=(0, 1))

    def build_constraints(self):
        """Build all the constraints."""
        if self.pb_number == 1:
            self.mdl.clust_assign = pe.Constraint(
                self.mdl.C, rule=clust_assign_
            )
        elif self.pb_number == 2:
            self.mdl.capacity = pe.Constraint(self.mdl.N, self.mdl.T,
                                              rule=capacity_)
            self.mdl.open_node = pe.Constraint(self.mdl.C, self.mdl.N,
                                               rule=open_node_)
            self.mdl.assignment = pe.Constraint(self.mdl.C,
                                                rule=assignment_)

    def build_objective(self):
        """Build the objective."""
        if self.pb_number == 1:
            self.mdl.obj = pe.Objective(
                rule=min_dissim_, sense=pe.minimize
            )
        elif self.pb_number == 2:
            self.mdl.obj = pe.Objective(
                rule=open_nodes_, sense=pe.minimize)

    def create_data(self,
                    df_indiv, dict_id_c,
                    df_host_meta,
                    nb_clusters
                    ):
        """Create data from dataframe."""
        if self.pb_number == 1:
            self.data = {None: {
                'c': {None: df_indiv[it.indiv_field].nunique()},
                'C': {None: list(dict_id_c.keys())},
                'k': {None: nb_clusters},
                'K': {None: range(nb_clusters)},
            }}
        elif self.pb_number == 2:
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

    def write_infile(self):
        """Write the problem in LP file."""
        if self.pb_number == 1:
            self.instance_model.write('./py_clustering.lp')
        elif self.pb_number == 2:
            self.instance_model.write('./py_placement.lp')


def clust_assign_(mdl, container):
    """Express the assignment constraint."""
    return sum(mdl.y[container, cluster] for cluster in mdl.K) == 1


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


def min_dissim_(mdl):
    """Express the within clusters dissimilarities."""
    return sum([
        mdl.u[(i, j)] * mdl.w[(i, j)] for i in mdl.C for j in mdl.C if i < j
    ])
