"""
=========
cots small_pyomo
=========

One model instance, made as an example alternative to the
'model_small_cplex' example made with CPLEX.
"""

from itertools import product as prod

from typing import Dict, List

import pandas as pd

from pyomo import environ as pe

import numpy as np

from . import init as it


class Model:
    """
    Class with a very small minimalist use of pyomo (for translation).

    Attributes :
    - pb_number : problem type (clustering or placement)
    - df_indiv : DataFrame describing containers data
    - dict_id_c : int id for containers with matching string id
    - dict_id_n : int id for nodes with matching string id
    - df_host_meta : DataFrame describing node info (capacity)
    - nb_clusters : number of clusters for clustering
    - w : similarity matrix for clustering
    - sol_u : adjacency matrix from current clustering solution
    - sol_v : adjacency matric from current placement solution
    """

    def __init__(self, pb_number: int,
                 df_indiv: pd.DataFrame, dict_id_c: Dict,
                 dict_id_n: Dict = None, df_host_meta: pd.DataFrame = None,
                 nb_clusters: int = None, w: np.array = None, dv: np.array = None,
                 sol_u: np.array = None, sol_v: np.array = None
                 ):
        """Initialize Pyomo model with data in Instance."""
        print('Building of pyomo model ...')

        # Which problem we build :
        # - 1 = only clustering
        # - 2 = placement from clustering
        self.pb_number = pb_number

        # Build the basis object "Model"
        self.mdl = pe.AbstractModel()

        # Prepare the sets and parameters
        self.build_parameters(w, dv, sol_u, sol_v)

        # Build decision variables
        self.build_variables()

        # Build constraints of the problem
        self.build_constraints()
        self.add_mustLink()

        # Build the objective function
        self.build_objective()

        # Put data in attribute
        # self.create_data(df_indiv, dict_id_c,
        #                  dict_id_n, df_host_meta,
        #                  nb_clusters
        #                  )
        self.create_data(df_indiv, dict_id_c,
                         df_host_meta,
                         nb_clusters
                         )

        # Create the instance by feeding the model with the data
        self.instance_model = self.mdl.create_instance(self.data)
        self.instance_model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
        

    def build_parameters(self, w, dv, u, v):
        """Build all Params and Sets."""
        # number of containers
        self.mdl.c = pe.Param(within=pe.NonNegativeIntegers)
        # set of containers
        self.mdl.C = pe.Set(dimen=1)
        # current clustering solution
        sol_u_d = dict(
            ((j, i), u[i][j]) for i,j in prod(range(len(u)),range(len(u[0])))
        )
        self.mdl.sol_u = pe.Param(self.mdl.C, self.mdl.C,
                                  initialize=sol_u_d)

        # clustering case
        if self.pb_number == 1:
            # number of clusters
            self.mdl.k = pe.Param(within=pe.NonNegativeIntegers)
            # set of clusters
            self.mdl.K = pe.Set(dimen=1)
            # distances
            w_d = dict(
                ((j, i), w[i][j]) for i,j in prod(range(len(w)),range(len(w[0])))
            )
            self.mdl.w = pe.Param(self.mdl.C, self.mdl.C,
                                  initialize=w_d)
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
            # set of containers applied to container usage problem
            self.mdl.Ccons = pe.Set(dimen=1)
            # containers usage
            self.mdl.cons = pe.Param(self.mdl.Ccons, self.mdl.T)
            # dv matrix for distance placement
            dv_d = dict(
                ((j, i), dv[i][j]) for i,j in prod(range(len(dv)),range(len(dv[0])))
            )
            self.mdl.dv = pe.Param(self.mdl.C, self.mdl.C,
                                      initialize=dv_d)
            # current placement solution
            sol_v_d = dict(
                ((j, i), v[i][j]) for i,j in prod(range(len(v)),range(len(v[0])))
            )
            self.mdl.sol_v = pe.Param(self.mdl.C, self.mdl.C,
                                      initialize=sol_v_d)

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
            self.mdl.open_cluster = pe.Constraint(
                self.mdl.C, self.mdl.K,
                rule=open_cluster_
            )
            self.mdl.max_clusters = pe.Constraint(
                rule=open_clusters_
            )
        elif self.pb_number == 2:
            self.mdl.capacity = pe.Constraint(
                self.mdl.N, self.mdl.T,
                rule=capacity_)
            self.mdl.open_node = pe.Constraint(
                self.mdl.C, self.mdl.N,
                rule=open_node_)
            self.mdl.assignment = pe.Constraint(
                self.mdl.C,
                rule=assignment_)

    def add_mustLink(self):
        """Add mustLink constraints for fixing solution."""
        if self.pb_number == 1:
            self.mdl.must_link_c = pe.Constraint(
                self.mdl.C, self.mdl.C,
                rule=must_link_c_
            )
        if self.pb_number == 2:
            self.mdl.must_link_n = pe.Constraint(
                self.mdl.C, self.mdl.C,
                rule=must_link_n_
            )

    def build_objective(self):
        """Build the objective."""
        if self.pb_number == 1:
            self.mdl.obj = pe.Objective(
                rule=min_dissim_, sense=pe.minimize
            )
        elif self.pb_number == 2:
            self.mdl.obj = pe.Objective(
                rule=min_coloc_cluster_, sense=pe.minimize)

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
                'C': {None: list(dict_id_c.keys())},
                'Ccons': {None: df_indiv[it.indiv_field].unique().tolist()},
                'cap': self.cap,
                'T': {None: df_indiv[it.tick_field].unique().tolist()},
                'cons': self.cons,
            }}

    def conso_n_t(self, mdl, node, t):
        """Express the total consumption of node at time t."""
        return sum(
            mdl.x[cont, node] * self.cons[contC][t]
            for cont,contC in zip(mdl.C, mdl.Ccons))

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

    def solve(self, solver='glpk'):
        """Solve the model using a specific solver."""
        opt = pe.SolverFactory(solver)
        opt.solve(self.instance_model, tee=True)
        # self.instance_model.display()
        # TODO verbose option ?
        print(pe.value(self.instance_model.obj))


def clust_assign_(mdl, container):
    """Express the assignment constraint."""
    return sum(mdl.y[container, cluster] for cluster in mdl.K) == 1


def capacity_(mdl, node, time):
    """Express the capacity constraints."""
    return (sum(
        mdl.x[i, node] * mdl.cons[j, time] for i,j in zip(mdl.C, mdl.Ccons)
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
    return sum(mdl.a[m] for m in mdl.N)


def open_cluster_(mdl, container, cluster):
    """Express the opening cluster constraint."""
    return mdl.y[container, cluster] <= mdl.b[cluster]


def open_clusters_(mdl):
    """Express the numbers of open clusters."""
    return sum(mdl.b[k] for k in mdl.K) <= mdl.k


def must_link_c_(mdl, i, j):
    """Express the clustering mustlink constraint."""
    if mdl.sol_u[(i, j)]:
        return mdl.u[(i, j)] == 1
    else:
        return pe.Constraint.Skip

def must_link_n_(mdl, i, j):
    """Express the placement mustlink constraint."""
    if mdl.sol_v[(i, j)]:
        return mdl.v[(i, j)] == 1
    else:
        return pe.Constraint.Skip


def min_dissim_(mdl):
    """Express the within clusters dissimilarities."""
    return sum([
        mdl.u[(i, j)] * mdl.w[(i, j)] for i,j in prod(mdl.C, mdl.C) if i < j
    ])


def min_coloc_cluster_(mdl: pe.AbstractModel):
    """Express the placement minimization objective from clustering."""
    return sum([
        mdl.sol_u[(i, j)] * mdl.v[(i, j)] for i,j in prod(mdl.C, mdl.C) if i < j
    ]) + sum([
        (1 - mdl.sol_u[(i, j)]) * mdl.v[(i, j)] * mdl.dv[(i, j)] for i,j in prod(mdl.C, mdl.C) if i < j
    ])


def fill_dual_values(my_mdl: Model):
    """Fill dual values from specific constraints."""
    dual_values = {}
    #TODO generalize with constraints in variables ?
    # Clustering case
    if my_mdl.pb_number == 1:
        for index_c in my_mdl.instance_model.must_link_c:
            dual_values[index_c] = my_mdl.instance_model.dual[
                my_mdl.instance_model.must_link_c[index_c]
            ]
    # Placement case
    if my_mdl.pb_number == 2:
        for index_c in my_mdl.instance_model.must_link_n:
            dual_values[index_c] = my_mdl.instance_model.dual[
                my_mdl.instance_model.must_link_n[index_c]
            ]
    return dual_values