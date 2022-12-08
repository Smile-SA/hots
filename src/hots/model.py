"""
=========
hots model
=========

Define the optimization models we have, with its objective, constraints,
variables, and build it from the ``Instance``. Provide all optimization models
related methods.
The optimization model description is based on Pyomo.
"""

from itertools import product as prod
from typing import Dict, List, Tuple

import networkx as nx

import numpy as np

import pandas as pd

from pyomo import environ as pe

from . import init as it
from .clustering import get_far_container


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
        sol_u_d = (
            ((j, i), u[i][j]) for i, j in prod(range(len(u)), range(len(u[0])))
        )
        self.mdl.sol_u = pe.Param(self.mdl.C, self.mdl.C,
                                  initialize=sol_u_d, mutable=True)

        # clustering case
        if self.pb_number == 1:
            # number of clusters
            self.mdl.k = pe.Param(within=pe.NonNegativeIntegers)
            # set of clusters
            self.mdl.K = pe.Set(dimen=1)
            # distances
            w_d = (
                ((j, i), w[i][j]) for i, j in prod(range(len(w)), range(len(w[0])))
            )
            self.mdl.w = pe.Param(self.mdl.C, self.mdl.C,
                                  initialize=w_d, mutable=True)
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
            dv_d = (
                ((j, i), dv[i][j]) for i, j in prod(range(len(dv)), range(len(dv[0])))
            )
            self.mdl.dv = pe.Param(
                self.mdl.C, self.mdl.C, initialize=dv_d, mutable=True)
            # current placement solution
            sol_v_d = (
                ((j, i), v[i][j]) for i, j in prod(range(len(v)), range(len(v[0])))
            )
            self.mdl.sol_v = pe.Param(self.mdl.C, self.mdl.C,
                                      initialize=sol_v_d, mutable=True)

    def build_variables(self):
        """Build all model variables."""
        if self.pb_number == 1:
            # Variables Containers x Clusters
            self.mdl.y = pe.Var(self.mdl.C, self.mdl.K,
                                domain=pe.NonNegativeReals,
                                bounds=(0, 1),
                                initialize=0)
            # Variables Containers x Containers
            self.mdl.u = pe.Var(self.mdl.C, self.mdl.C,
                                domain=pe.NonNegativeReals,
                                bounds=(0, 1),
                                initialize=0)
            # Variables Clusters
            self.mdl.b = pe.Var(self.mdl.K,
                                domain=pe.NonNegativeReals,
                                bounds=(0, 1),
                                initialize=0)
        elif self.pb_number == 2:
            # Variables Containers x Nodes
            self.mdl.x = pe.Var(self.mdl.C, self.mdl.N,
                                domain=pe.NonNegativeReals,
                                bounds=(0, 1),
                                initialize=0)
            # Variables Nodes
            self.mdl.a = pe.Var(self.mdl.N,
                                domain=pe.NonNegativeReals,
                                bounds=(0, 1),
                                initialize=0)
            # Variables Containers x Containers
            self.mdl.v = pe.Var(self.mdl.C, self.mdl.C,
                                domain=pe.NonNegativeReals,
                                bounds=(0, 1),
                                initialize=0)

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

    def add_mustlink_instance(self):
        """Add mustLink constraints for fixing solution."""
        if self.pb_number == 1:
            self.instance_model.must_link_c = pe.Constraint(
                self.instance_model.C, self.instance_model.C,
                rule=must_link_c_
            )
        if self.pb_number == 2:
            self.instance_model.must_link_n = pe.Constraint(
                self.instance_model.C, self.instance_model.C,
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
            for n, n_data in df_host_meta.groupby(it.host_field):
                self.cap.update({n: n_data['cpu'].values[0]})
            self.cons = {}
            df_indiv.reset_index(drop=True, inplace=True)
            for key, c_data in df_indiv.groupby([it.indiv_field, it.tick_field]):
                self.cons.update({key: c_data['cpu'].values[0]})

            self.data = {None: {
                'n': {None: df_host_meta[it.host_field].nunique()},
                'c': {None: df_indiv[it.indiv_field].nunique()},
                't': {None: df_indiv[it.tick_field].nunique()},
                'N': {None: df_host_meta[it.host_field].unique().tolist()},
                'C': {None: list(dict_id_c.keys())},
                'Ccons': {None: df_indiv[it.indiv_field].unique().tolist()},
                'cap': self.cap,
                'T': {None: df_indiv[it.tick_field].unique().tolist()},
                'cons': self.cons,
            }}

    def conso_n_t(self, mdl, node, t):
        """Express the total consumption of node at time t."""
        return sum(
            mdl.x[cont, node] * self.cons[cont_c][t]
            for cont, cont_c in zip(mdl.C, mdl.Ccons))

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
        # print(pe.value(self.instance_model.obj))

    # TODO generalize with others constraints than mustlink
    def update_adjacency_clust_constraints(self, u):
        """Update constraints fixing u variables from new adjacency matrix (clustering)."""
        self.instance_model.del_component(self.instance_model.must_link_c)
        self.instance_model.del_component(self.instance_model.must_link_c_index)
        self.update_sol_u(u)
        self.add_mustLink_instance()

    def update_sol_u(self, u):
        """Update directly the sol_u param in instance from new u matrix."""
        for i, j in prod(range(len(u)), range(len(u[0]))):
            self.instance_model.sol_u[(i, j)] = u[i][j]

    # TODO generalize with others constraints than mustlink
    def update_adjacency_place_constraints(self, v):
        """Update constraints fixing u variables from new adjacency matrix (placement)."""
        self.instance_model.del_component(self.instance_model.must_link_n)
        self.instance_model.del_component(self.instance_model.must_link_n_index)
        self.update_sol_v(v)
        self.add_mustLink_instance()

    def update_sol_v(self, v):
        """Update directly the sol_v param in instance from new v matrix."""
        for i, j in prod(range(len(v)), range(len(v[0]))):
            self.instance_model.sol_v[(i, j)] = v[i][j]

    def update_obj_clustering(self, w):
        """Update the objective for clustering with new w matrix."""
        self.update_w(w)
        self.instance_model.obj = sum([
            self.instance_model.u[(i, j)] * self.instance_model.w[(i, j)] for i, j in prod(
                self.instance_model.C, self.instance_model.C
            ) if i < j
        ])

    def update_w(self, w):
        """Update directly the w param in instance from new w matrix."""
        for i, j in prod(range(len(w)), range(len(w[0]))):
            self.instance_model.w[(i, j)] = w[i][j]

    def update_obj_place(self, dv):
        """Update the objective for placement with new dv matrix."""
        self.update_dv(dv)
        self.instance_model.obj = sum([
            self.instance_model.sol_u[(i, j)] * self.instance_model.v[(i, j)] for i, j in prod(
                self.instance_model.C, self.instance_model.C
            ) if i < j
        ]) + sum([
            (1 - self.instance_model.sol_u[(i, j)]) * (
                self.instance_model.v[(i, j)] * self.instance_model.dv[(i, j)]
            ) for i, j in prod(self.instance_model.C, self.instance_model.C) if i < j
        ])

    def update_dv(self, dv):
        """Update directly the dv param in instance from new dv matrix."""
        for i, j in prod(range(len(dv)), range(len(dv[0]))):
            self.instance_model.dv[(i, j)] = dv[i][j]

    # TODO to finish
    def update_instance(self, df_indiv):
        """Update the model instance from new data."""
        # Clear data and constraints
        for c in self.instance_model.component_objects(pe.Constraint):
            self.instance_model.del_component(c)
        # Re-create data

        # Re-create constraints


def clust_assign_(mdl, container):
    """Express the assignment constraint."""
    return sum(mdl.y[container, cluster] for cluster in mdl.K) == 1


def capacity_(mdl, node, time):
    """Express the capacity constraints."""
    return (sum(
        mdl.x[i, node] * mdl.cons[j, time] for i, j in zip(mdl.C, mdl.Ccons)
    ) <= mdl.cap[node])


def open_node_(mdl, container, node):
    """Express the opening node constraint."""
    return mdl.x[container, node] <= mdl.a[node]


def assignment_(mdl, container):
    """Express the assignment constraint."""
    return sum(mdl.x[container, node] for node in mdl.N) == 1


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
    uu = mdl.sol_u[(i, j)].value
    if uu == 1:
        return mdl.u[(i, j)] == 1
    else:
        return pe.Constraint.Skip


def must_link_n_(mdl, i, j):
    """Express the placement mustlink constraint."""
    vv = mdl.sol_v[(i, j)].value
    if vv == 1:
        return mdl.v[(i, j)] == 1
    else:
        return pe.Constraint.Skip


def min_dissim_(mdl):
    """Express the within clusters dissimilarities."""
    return sum([
        mdl.u[(i, j)] * mdl.w[(i, j)] for i, j in prod(mdl.C, mdl.C) if i < j
    ])


def min_coloc_cluster_(mdl: pe.AbstractModel):
    """Express the placement minimization objective from clustering."""
    return sum([
        mdl.sol_u[(i, j)] * mdl.v[(i, j)] for i, j in prod(mdl.C, mdl.C) if i < j
    ]) + sum([(
            (1 - mdl.sol_u[(i, j)]) * mdl.v[(i, j)] * mdl.dv[(i, j)]
    ) for i, j in prod(mdl.C, mdl.C) if i < j])


def fill_dual_values(my_mdl: Model):
    """Fill dual values from specific constraints."""
    dual_values = {}
    # TODO generalize with constraints in variables ?
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


def get_conflict_graph(my_mdl: Model, constraints_dual_values: Dict, tol: float):
    """Build conflict graph from comapring dual variables."""
    conflict_graph = nx.Graph()
    if my_mdl.pb_number == 1:
        for index_c in my_mdl.instance_model.must_link_c:
            if (index_c in constraints_dual_values) and (
                constraints_dual_values[index_c] > 0.0
            ):
                if (my_mdl.instance_model.dual[
                    my_mdl.instance_model.must_link_c[index_c]] > (
                    constraints_dual_values[index_c]
                    + tol * constraints_dual_values[index_c])) or (
                        my_mdl.instance_model.dual[
                            my_mdl.instance_model.must_link_c[index_c]
                        ] > tol * pe.value(my_mdl.instance_model.obj)
                ):
                    conflict_graph.add_edge(
                        index_c[0], index_c[1],
                        weight=my_mdl.instance_model.dual[
                            my_mdl.instance_model.must_link_c[index_c]])
    elif my_mdl.pb_number == 2:
        for index_c in my_mdl.instance_model.must_link_n:
            if (index_c in constraints_dual_values) and (
                constraints_dual_values[index_c] > 0.0
            ):
                if (my_mdl.instance_model.dual[
                    my_mdl.instance_model.must_link_n[index_c]] > (
                    constraints_dual_values[index_c]
                    + tol * constraints_dual_values[index_c])) or (
                        my_mdl.instance_model.dual[
                            my_mdl.instance_model.must_link_n[index_c]
                        ] > tol * pe.value(my_mdl.instance_model.obj)
                ):
                    conflict_graph.add_edge(
                        index_c[0], index_c[1],
                        weight=my_mdl.instance_model.dual[
                            my_mdl.instance_model.must_link_n[index_c]])
    return conflict_graph


def get_moving_containers_clust(my_mdl: Model, constraints_dual_values: Dict,
                                tol: float, tol_move: float, nb_containers: int,
                                dict_id_c: Dict, df_clust: pd.DataFrame, profiles: np.array
                                ) -> Tuple[List, int, int, int, int]:
    """Get the list of moving containers from constraints dual values."""
    mvg_containers = []
    conflict_graph = get_conflict_graph(my_mdl, constraints_dual_values, tol)

    graph_nodes = conflict_graph.number_of_nodes()
    graph_edges = conflict_graph.number_of_edges()
    list_indivs = sorted(conflict_graph.degree, key=lambda x: x[1], reverse=True)
    if len(list_indivs) == 0:
        max_deg = 0
        mean_deg = 0
    else:
        max_deg = list_indivs[0][1]
        mean_deg = sum(
            [deg for (node, deg) in list_indivs]
        ) / float(len(conflict_graph))
    while len(list_indivs) > 1:
        (indiv, occur) = list_indivs[0]
        if occur > 1:
            it = 1
            w_deg = conflict_graph.degree(indiv, weight='weight')
            (indiv_bis, occur_bis) = list_indivs[it]
            while occur_bis == occur:
                w_deg_bis = conflict_graph.degree(indiv_bis, weight='weight')
                if w_deg < w_deg_bis:
                    w_deg = w_deg_bis
                    indiv = indiv_bis
                it += 1
                if it >= len(list_indivs):
                    break
                else:
                    (indiv_bis, occur_bis) = list_indivs[it]
            mvg_containers.append(dict_id_c[int(indiv)])
            conflict_graph.remove_node(indiv)
        else:
            other_indiv = list(conflict_graph.edges(indiv))[0][1]
            if other_indiv == indiv:
                other_indiv = list(conflict_graph.edges(indiv))[0][0]
            mvg_indiv = get_far_container(
                dict_id_c[int(indiv)],
                dict_id_c[int(other_indiv)],
                df_clust, profiles
            )
            mvg_containers.append(mvg_indiv)
            conflict_graph.remove_node(indiv)
            conflict_graph.remove_node(other_indiv)
        if len(mvg_containers) >= (nb_containers * tol_move):
            break
        conflict_graph.remove_nodes_from(list(nx.isolates(conflict_graph)))
        list_indivs = sorted(conflict_graph.degree, key=lambda x: x[1], reverse=True)

    return (mvg_containers, graph_nodes, graph_edges,
            max_deg, mean_deg)


# TODO to improve : very low dual values can change easily
# TODO choose container by most changing profile ?
def get_moving_containers_place(my_mdl: Model, constraints_dual_values: Dict,
                                tol: float, tol_move: float, nb_containers: int,
                                working_df: pd.DataFrame, dict_id_c: Dict
                                ) -> Tuple[List, int, int, int, int]:
    """Get the list of moving containers from constraints dual values."""
    mvg_containers = []
    conflict_graph = get_conflict_graph(my_mdl, constraints_dual_values, tol)

    graph_nodes = conflict_graph.number_of_nodes()
    graph_edges = conflict_graph.number_of_edges()
    list_indivs = sorted(conflict_graph.degree, key=lambda x: x[1], reverse=True)
    if len(list_indivs) == 0:
        max_deg = 0
        mean_deg = 0
    else:
        max_deg = list_indivs[0][1]
        mean_deg = sum(
            [deg for (node, deg) in list_indivs]
        ) / float(len(conflict_graph))
    while len(list_indivs) > 1:
        (indiv, occur) = list_indivs[0]
        if occur > 1:
            it = 1
            w_deg = conflict_graph.degree(indiv, weight='weight')
            (indiv_bis, occur_bis) = list_indivs[it]
            while occur_bis == occur:
                w_deg_bis = conflict_graph.degree(indiv_bis, weight='weight')
                if w_deg < w_deg_bis:
                    w_deg = w_deg_bis
                    indiv = indiv_bis
                it += 1
                if it >= len(list_indivs):
                    break
                else:
                    (indiv_bis, occur_bis) = list_indivs[it]
            mvg_containers.append(int(indiv))
            conflict_graph.remove_node(indiv)
        else:
            other_indiv = list(conflict_graph.edges(indiv))[0][1]
            if other_indiv == indiv:
                other_indiv = list(conflict_graph.edges(indiv))[0][0]
            mvg_indiv = get_container_tomove(
                dict_id_c[int(indiv)],
                dict_id_c[int(other_indiv)],
                working_df
            )
            int_indiv = [k for k, v in dict_id_c.items() if v == mvg_indiv][0]
            mvg_containers.append(int(int_indiv))
            conflict_graph.remove_node(indiv)
            conflict_graph.remove_node(other_indiv)
        if len(mvg_containers) >= (nb_containers * tol_move):
            break
        conflict_graph.remove_nodes_from(list(nx.isolates(conflict_graph)))
        list_indivs = sorted(conflict_graph.degree, key=lambda x: x[1], reverse=True)

    return (mvg_containers, graph_nodes, graph_edges,
            max_deg, mean_deg)


def get_container_tomove(c1: int, c2: int, working_df: pd.DataFrame) -> int:
    """Get the container we want to move between c1 and c2."""
    node = working_df.loc[
        working_df[it.indiv_field] == c1][it.host_field].to_numpy()[0]
    node_data = working_df.loc[
        working_df[it.host_field] == node].groupby(
            working_df[it.tick_field]
    )[it.metrics[0]].sum().to_numpy()
    c1_cons = working_df.loc[
        working_df[it.indiv_field] == c1
    ][it.metrics[0]].to_numpy()
    c2_cons = working_df.loc[
        working_df[it.indiv_field] == c2
    ][it.metrics[0]].to_numpy()
    if (node_data - c1_cons).var() < (node_data - c2_cons).var():
        return c1
    else:
        return c2


def get_obj_value_host(df_host: pd.DataFrame,
                       t_min: int = None,
                       t_max: int = None) -> Tuple[int, float]:
    """Get objectives value of current solution."""
    t_min = t_min or df_host[it.tick_field].min()
    t_max = t_max or df_host[it.tick_field].max()
    df_host = df_host[
        (df_host[it.tick_field] >= t_min)
        & (df_host[it.tick_field] <= t_max)]
    df_host.reset_index(drop=True, inplace=True)
    c2 = 0.0
    nb_nodes = 0
    for n, n_data in df_host.groupby(
            df_host[it.host_field]):
        if n_data[it.metrics[0]].mean() > 1e-6:
            nb_nodes += 1
            c2_n = n_data[it.metrics[0]].max() - n_data[it.metrics[0]].min()
            if c2_n > c2:
                c2 = c2_n
    return (nb_nodes, c2)


def get_obj_value_indivs(
    df_indiv: pd.DataFrame,
    t_min: int = None,
    t_max: int = None
) -> Tuple[int, float]:
    """Get objective value of current solution (max delta)."""
    t_min = t_min or df_indiv[it.tick_field].min()
    t_max = t_max or df_indiv[it.tick_field].max()
    df_indiv = df_indiv[
        (df_indiv[it.tick_field] >= t_min)
        & (df_indiv[it.tick_field] <= t_max)]
    df_indiv.reset_index(drop=True, inplace=True)
    obj_val = 0.0
    for n, n_data in df_indiv.groupby(it.host_field):
        max_n = 0.0
        min_n = -1.0
        for t, nt_data in n_data.groupby(it.tick_field):
            if nt_data[it.metrics[0]].sum() > max_n:
                max_n = nt_data[it.metrics[0]].sum()
            if (nt_data[it.metrics[0]].sum() < min_n) or (
                    min_n < 0.0):
                min_n = nt_data[it.metrics[0]].sum()
        delta_n = max_n - min_n
        if obj_val < delta_n:
            obj_val = delta_n
    return (df_indiv[it.host_field].nunique(), obj_val)
