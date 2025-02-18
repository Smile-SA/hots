"""
Define the optimization models.

Describing its objective, constraints, variables,
and build it from the ``Instance``.
Provide all optimization models related methods.
The optimization model description is based on Pyomo.
"""

from itertools import product as prod

import networkx as nx

from pyomo import environ as pe

from . import init as it
from .clustering import get_far_container
from pyomo.common.errors import ApplicationError
from pyomo.opt import TerminationCondition
from pyomo.environ import Constraint


class Model:
    """
    Class holding the optimization models creation.

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

    def __init__(
        self, pb_number, df_indiv, metric, dict_id_c, df_host_meta=None,
        nb_clusters=None, w=None, dv=None, sol_u=None, sol_v=None, verbose=False
    ):
        """Initialize Pyomo model with data in Instance.

        :param pb_number: _description_
        :type pb_number: int
        :param df_indiv: _description_
        :type df_indiv: pd.DataFrame
        :param metric: _description_
        :type metric: str
        :param dict_id_c: _description_
        :type dict_id_c: Dict
        :param df_host_meta: _description_, defaults to None
        :type df_host_meta: pd.DataFrame, optional
        :param nb_clusters: _description_, defaults to None
        :type nb_clusters: int, optional
        :param w: _description_, defaults to None
        :type w: np.array, optional
        :param dv: _description_, defaults to None
        :type dv: np.array, optional
        :param sol_u: _description_, defaults to None
        :type sol_u: np.array, optional
        :param sol_v: _description_, defaults to None
        :type sol_v: np.array, optional
        """
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
        self.add_mustlink()

        # Build the objective function
        self.build_objective()

        # Put data in attribute
        self.create_data(df_indiv, metric, dict_id_c,
                         df_host_meta,
                         nb_clusters
                         )

        # Create the instance by feeding the model with the data
        self.instance_model = self.mdl.create_instance(self.data)
        if verbose:
            self.write_infile()
        self.instance_model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    def build_parameters(self, w, dv, u, v):
        """Build all Params and Sets.

        :param w: _description_
        :type w: np.array
        :param dv: _description_
        :type dv: np.array
        :param u: _description_
        :type u: np.array
        :param v: _description_
        :type v: np.array
        """
        # number of containers
        self.mdl.c = pe.Param(within=pe.NonNegativeIntegers, mutable=True)
        # set of containers
        self.mdl.C = pe.Set(dimen=1)
        # current clustering solution
        sol_u_d = {
            (j, i): u[i][j] for i, j in prod(range(len(u)), range(len(u[0])))
        }
        self.mdl.sol_u = pe.Param(self.mdl.C, self.mdl.C,
                                  initialize=sol_u_d, mutable=True)

        # clustering case
        if self.pb_number == 1:
            # number of clusters
            self.mdl.k = pe.Param(within=pe.NonNegativeIntegers)
            # set of clusters
            self.mdl.K = pe.Set(dimen=1)
            # distances
            w_d = {
                (j, i): w[i][j] for i, j in prod(
                    range(len(w)), range(len(w[0])))
            }
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
            self.mdl.cons = pe.Param(self.mdl.Ccons, self.mdl.T, mutable=True)
            # dv matrix for distance placement
            dv_d = {
                (j, i): dv[i][j] for i, j in prod(
                    range(len(dv)), range(len(dv[0])))
            }
            self.mdl.dv = pe.Param(
                self.mdl.C, self.mdl.C, initialize=dv_d, mutable=True)
            # current placement solution
            sol_v_d = {
                (j, i): v[i][j] for i, j in prod(
                    range(len(v)), range(len(v[0])))
            }
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

    def add_mustlink(self):
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

    def create_data(self, df_indiv, metric, dict_id_c,
                    df_host_meta, nb_clusters):
        """Create data from dataframe.

        :param df_indiv: _description_
        :type df_indiv: _type_
        :param metric: _description_
        :type metric: str
        :param dict_id_c: _description_
        :type dict_id_c: _type_
        :param df_host_meta: _description_
        :type df_host_meta: _type_
        :param nb_clusters: _description_
        :type nb_clusters: _type_
        """
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
                self.cap.update({n: n_data[metric].values[0]})
            self.cons = {}
            df_indiv.reset_index(drop=True, inplace=True)
            for key, c_data in df_indiv.groupby(
                [it.indiv_field, it.tick_field]
            ):
                self.cons.update({key: c_data[metric].values[0]})
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
        """Express the total consumption of node at time t.

        :param mdl: _description_
        :type mdl: _type_
        :param node: _description_
        :type node: _type_
        :param t: _description_
        :type t: _type_
        :return: _description_
        :rtype: _type_
        """
        return sum(
            mdl.x[cont, node] * self.cons[cont_c][t]
            for cont, cont_c in zip(mdl.C, mdl.Ccons))

    def mean(self, mdl, node):
        """Express the mean consumption of node.

        :param mdl: _description_
        :type mdl: _type_
        :param node: _description_
        :type node: _type_
        :return: _description_
        :rtype: _type_
        """
        return (sum(
            self.conso_n_t(node, t) for t in mdl.T
        ) / mdl.t)

    def write_infile(self):
        """Write the problem in LP file."""
        if self.pb_number == 1:
            self.instance_model.write(
                './py_clustering.lp',
                io_options={'symbolic_solver_labels': True}
            )
        elif self.pb_number == 2:
            self.instance_model.write(
                './py_placement.lp',
                io_options={'symbolic_solver_labels': True}
            )

    def solve(self, solver='glpk', verbose=False):
        """Solve the model using a specific solver.

        :param solver: The solver to use to solve the problem.
        :type solver: str
        :param verbose: Enable / disable logs during solve process.
        :type verbose: bool
        """
        opt = pe.SolverFactory(solver)
        results = None
        try:
            results = opt.solve(self.instance_model, tee=verbose)
        except ApplicationError as e:
            print(f"Solver error: {e}")

        if verbose:
            for c in self.instance_model.component_objects(Constraint, active=True):
                print(f"\n🔍 Checking Constraint: {c.name}")

                for index in c:
                    lhs = c[index].body()
                    upper = c[index].upper
                    lower = c[index].lower

                    if upper is not None and lhs > upper:
                        print(f"  ❌ Constraint {c.name}[{index}] is violated!")
                        print(f"     Expected: LHS ≤ {upper}")
                        print(f"     Got: LHS = {lhs} (TOO HIGH!)")

                    if lower is not None and lhs < lower:
                        print(f"  ❌ Constraint {c.name}[{index}] is violated!")
                        print(f"     Expected: LHS ≥ {lower}")
                        print(f"     Got: LHS = {lhs} (TOO LOW!)")

                    if (upper is None or lhs <= upper) and (lower is None or lhs >= lower):
                        print(f"  ✅ Constraint {c.name}[{index}] holds. (LHS = {lhs})")

            print("---------------------------------------------------------------------")
            
            if results.solver.termination_condition == TerminationCondition.infeasible:
                print("Problem is infeasible!")
            elif results.solver.termination_condition == TerminationCondition.unbounded:
                print("Problem is unbounded!")
            elif results.solver.termination_condition == TerminationCondition.optimal:
                print("Solution found.")
            else:
                print(f"Solver status: {results.solver.termination_condition}")
        
            print('---------------------------------------------------------------------')
            print(f'Objective function value: {pe.value(self.instance_model.obj)}')
        # self.instance_model.display()

    # TODO generalize with others constraints than mustlink
    def update_adjacency_clust_constraints(self, u):
        """Update constraints fixing u variables from new adjacency matrix.

        :param u: _description_
        :type u: _type_
        """
        # self.instance_model.pprint()
        # self.instance_model.must_link_c.pprint()
        # print(self.instance_model.must_link_c.is_indexed())
        self.instance_model.del_component(self.instance_model.must_link_c)
        # self.instance_model.del_component(self.instance_model.must_link_c_index)
        self.update_sol_u(u)
        self.add_mustlink_instance()

    def update_sol_u(self, u):
        """Update directly the sol_u param in instance from new u matrix.

        :param u: _description_
        :type u: _type_
        """
        for i, j in prod(range(len(u)), range(len(u[0]))):
            self.instance_model.sol_u[(i, j)] = u[i][j]

    # TODO generalize with others constraints than mustlink
    def update_adjacency_place_constraints(self, v):
        """Update constraints fixing v variables from new adjacency matrix.

        :param v: _description_
        :type v: _type_
        """
        self.instance_model.del_component(self.instance_model.must_link_n)
        # self.instance_model.del_component(self.instance_model.must_link_n_index)
        self.update_sol_v(v)
        self.add_mustlink_instance()

    def update_sol_v(self, v):
        """Update directly the sol_v param in instance from new v matrix.

        :param v: _description_
        :type v: _type_
        """
        for i, j in prod(range(len(v)), range(len(v[0]))):
            self.instance_model.sol_v[(i, j)] = v[i][j]

    def update_obj_clustering(self, w):
        """Update the objective for clustering with new w matrix.

        :param w: _description_
        :type w: _type_
        """
        self.update_w(w)
        self.instance_model.obj = sum([
            self.instance_model.u[(i, j)] * (
                self.instance_model.w[(i, j)]) for i, j in prod(
                self.instance_model.C, self.instance_model.C
            ) if i < j
        ])

    def update_w(self, w):
        """Update directly the w param in instance from new w matrix.

        :param w: _description_
        :type w: _type_
        """
        for i, j in prod(range(len(w)), range(len(w[0]))):
            self.instance_model.w[(i, j)] = w[i][j]

    def update_obj_place(self, dv):
        """Update the objective for placement with new dv matrix.

        :param dv: _description_
        :type dv: _type_
        """
        self.update_dv(dv)
        self.instance_model.obj = sum([
            self.instance_model.sol_u[(i, j)] * (
                self.instance_model.v[(i, j)]) for i, j in prod(
                self.instance_model.C, self.instance_model.C
            ) if i < j
        ]) + sum([
            (1 - self.instance_model.sol_u[(i, j)]) * (
                self.instance_model.v[(i, j)] * self.instance_model.dv[(i, j)]
            ) for i, j in prod(self.instance_model.C,
                               self.instance_model.C) if i < j
        ])

    def update_dv(self, dv):
        """Update directly the dv param in instance from new dv matrix.

        :param dv: _description_
        :type dv: _type_
        """
        for i, j in prod(range(len(dv)), range(len(dv[0]))):
            self.instance_model.dv[(i, j)] = dv[i][j]

    def update_size_model(self, df_indiv=None, w=None, u=None, dv=None, v=None, verbose=False):
        """Update the model instance based on new number of containers."""
        print('Updating model for ', self.pb_number)
        # new_containers = list(it.my_instance.dict_id_c.keys())
        # new_containers_names = list(it.my_instance.container_to_id.keys())

        del self.mdl
        self.mdl = pe.AbstractModel()

        # Prepare the sets and parameters
        self.build_parameters(w, dv, u, v)

        # Build decision variables
        self.build_variables()

        # Build constraints of the problem
        self.build_constraints()
        self.add_mustlink()

        # Build the objective function
        self.build_objective()

        # Put data in attribute
        self.create_data(
            df_indiv, it.metrics[0], it.my_instance.dict_id_c,
            it.my_instance.df_host_meta, it.my_instance.nb_clusters
        )

        self.instance_model = self.mdl.create_instance(self.data)
        if verbose:
            self.write_infile()
        self.instance_model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
        # Update the container set
        # self.instance_model.C.clear()
        # self.instance_model.C.update(new_containers)

        # Update parameters dynamically
        # self.instance_model.c = len(new_containers)

        # clustering case
        # if self.pb_number == 1:
        #     # Recompute the w dictionary with updated indices
        #     new_w_d = {
        #         (j, i): w[i][j]
        #         for i in range(len(w))
        #         for j in range(len(w[0]))
        #         if i in new_containers and j in new_containers
        #     }
        #     self.instance_model.w.store_values(new_w_d)

        #     # Recompute sol_u using the updated `u` matrix
        #     new_sol_u = {
        #         (j, i): u[i][j]
        #         for i in range(len(u))
        #         for j in range(len(u[0]))
        #         if i in new_containers and j in new_containers
        #     }
        #     self.instance_model.sol_u.store_values(new_sol_u)

        #     self.data = {None: {
        #         'c': {None: len(new_containers)},
        #         'C': {None: new_containers},
        #         'k': {None: it.my_instance.nb_clusters},
        #         'K': {None: range(it.my_instance.nb_clusters)},
        #     }}

        #     # Reconstruct variable to apply dependancies with parameters changes
        #     # self.instance_model.u.clear()
        #     # self.instance_model.u._constructed = False
        #     # self.instance_model.u.construct()
        #     # self.instance_model.y.clear()
        #     # self.instance_model.y._constructed = False
        #     # self.instance_model.y.construct()

        #     # self.instance_model.clust_assign.clear()
        #     # self.instance_model.clust_assign.construct()
        #     # self.instance_model.open_cluster.clear()
        #     # self.instance_model.open_cluster.construct()
        #     # self.instance_model.max_clusters.clear()
        #     # self.instance_model.max_clusters.construct()
        #     # self.instance_model.must_link_c.clear()
        #     # self.instance_model.must_link_c.construct()

        #     # self.instance_model.obj.clear()
        #     # self.instance_model.obj.construct()
        #     # TODO check other dependancies

        # # placement case
        # if self.pb_number == 2:
        #     self.instance_model.Ccons.clear()
        #     self.instance_model.Ccons.update(new_containers_names)
        #     self.instance_model.T.clear()
        #     self.instance_model.T.update(df_indiv[it.tick_field].unique().tolist())

        #     self.cons = {}
        #     df_indiv.reset_index(drop=True, inplace=True)
        #     for key, c_data in df_indiv.groupby(
        #         [it.indiv_field, it.tick_field]
        #     ):
        #         self.cons.update({key: c_data[it.metrics[0]].values[0]})
        #     self.instance_model.cons.clear()
        #     self.instance_model.cons.store_values(self.cons)

        #     new_dv_d = {
        #         (j, i): dv[i][j]
        #         for i in range(len(dv))
        #         for j in range(len(dv[0]))
        #         if i in new_containers and j in new_containers
        #     }
        #     self.instance_model.dv.store_values(new_dv_d)

        #     # Recompute sol_u using the updated `u` matrix
        #     new_sol_v = {
        #         (j, i): v[i][j]
        #         for i in range(len(v))
        #         for j in range(len(v[0]))
        #         if i in new_containers and j in new_containers
        #     }
        #     self.instance_model.sol_v.store_values(new_sol_v)

        #     # Reconstruct variable to apply dependancies with parameters changes
        #     self.instance_model.v.clear()
        #     self.instance_model.v._constructed = False
        #     self.instance_model.v.construct()
        #     self.instance_model.x.clear()
        #     self.instance_model.x._constructed = False
        #     self.instance_model.x.construct()
            # TODO check other dependancies


def clust_assign_(mdl, container):
    """Express the assignment constraint.

    :param mdl: _description_
    :type mdl: _type_
    :param container: _description_
    :type container: _type_
    :return: _description_
    :rtype: _type_
    """
    return sum(mdl.y[container, cluster] for cluster in mdl.K) == 1


def capacity_(mdl, node, time):
    """Express the capacity constraints.

    :param mdl: _description_
    :type mdl: _type_
    :param node: _description_
    :type node: _type_
    :param time: _description_
    :type time: _type_
    :return: _description_
    :rtype: _type_
    """
    return (sum(
        mdl.x[i, node] * mdl.cons[j, time] for i, j in zip(mdl.C, mdl.Ccons)
    ) <= mdl.cap[node])


def open_node_(mdl, container, node):
    """Express the opening node constraint.

    :param mdl: _description_
    :type mdl: _type_
    :param container: _description_
    :type container: _type_
    :param node: _description_
    :type node: _type_
    :return: _description_
    :rtype: _type_
    """
    return mdl.x[container, node] <= mdl.a[node]


def assignment_(mdl, container):
    """Express the assignment constraint.

    :param mdl: _description_
    :type mdl: _type_
    :param container: _description_
    :type container: _type_
    :return: _description_
    :rtype: _type_
    """
    return sum(mdl.x[container, node] for node in mdl.N) == 1


def open_nodes_(mdl):
    """Express the numbers of open nodes.

    :param mdl: _description_
    :type mdl: _type_
    :return: _description_
    :rtype: _type_
    """
    return sum(mdl.a[m] for m in mdl.N)


def open_cluster_(mdl, container, cluster):
    """Express the opening cluster constraint.

    :param mdl: _description_
    :type mdl: _type_
    :param container: _description_
    :type container: _type_
    :param cluster: _description_
    :type cluster: _type_
    :return: _description_
    :rtype: _type_
    """
    return mdl.y[container, cluster] <= mdl.b[cluster]


def open_clusters_(mdl):
    """Express the numbers of open clusters.

    :param mdl: _description_
    :type mdl: _type_
    :return: _description_
    :rtype: _type_
    """
    return sum(mdl.b[k] for k in mdl.K) <= mdl.k


def must_link_c_(mdl, i, j):
    """Express the clustering mustlink constraint.

    :param mdl: _description_
    :type mdl: _type_
    :param i: _description_
    :type i: _type_
    :param j: _description_
    :type j: _type_
    :return: _description_
    :rtype: _type_
    """
    uu = mdl.sol_u[(i, j)].value
    if uu == 1:
        return mdl.u[(i, j)] == 1
    else:
        return pe.Constraint.Skip


def must_link_n_(mdl, i, j):
    """Express the placement mustlink constraint.

    :param mdl: _description_
    :type mdl: _type_
    :param i: _description_
    :type i: _type_
    :param j: _description_
    :type j: _type_
    :return: _description_
    :rtype: _type_
    """
    vv = mdl.sol_v[(i, j)].value
    if vv == 1:
        return mdl.v[(i, j)] == 1
    else:
        return pe.Constraint.Skip


def min_dissim_(mdl):
    """Express the within clusters dissimilarities.

    :param mdl: _description_
    :type mdl: _type_
    :return: _description_
    :rtype: _type_
    """
    return sum([
        mdl.u[(i, j)] * mdl.w[(i, j)] for i, j in prod(mdl.C, mdl.C) if i < j
    ])


def min_coloc_cluster_(mdl):
    """Express the placement minimization objective from clustering.

    :param mdl: _description_
    :type mdl: pe.AbstractModel
    :return: _description_
    :rtype: _type_
    """
    return sum([
        mdl.sol_u[(i, j)] * mdl.v[(i, j)] for i, j in
        prod(mdl.C, mdl.C) if i < j
    ]) + sum([(
            (1 - mdl.sol_u[(i, j)]) * mdl.v[(i, j)] * mdl.dv[(i, j)]
    ) for i, j in prod(mdl.C, mdl.C) if i < j])


def fill_dual_values(my_mdl):
    """Fill dual values from specific constraints.

    :param my_mdl: _description_
    :type my_mdl: Model
    :return: _description_
    :rtype: Dict
    """
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


def get_conflict_graph_old(my_mdl, constraints_dual_values, tol):
    """Build conflict graph from comapring dual variables.

    :param my_mdl: _description_
    :type my_mdl: Model
    :param constraints_dual_values: _description_
    :type constraints_dual_values: Dict
    :param tol: _description_
    :type tol: float
    :return: _description_
    :rtype: nx.Graph
    """
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


def get_conflict_graph(my_mdl, constraints_dual_values, tol):
    """Build conflict graph from comapring dual variables.

    :param my_mdl: _description_
    :type my_mdl: Model
    :param constraints_dual_values: _description_
    :type constraints_dual_values: Dict
    :param tol: _description_
    :type tol: float
    :return: _description_
    :rtype: nx.Graph
    """
    # max_edges = len(my_mdl.instance_model.must_link_c)
    # conflict_graph = nx.Graph()

    # Store frequently used values
    tol_value = tol * pe.value(my_mdl.instance_model.obj)

    if my_mdl.pb_number == 1:
        max_edges = len(my_mdl.instance_model.must_link_c)
        conflict_graph = nx.Graph(capacity=max_edges)

        # Store frequently used values
        must_link_c = my_mdl.instance_model.must_link_c
        instance_model_dual = my_mdl.instance_model.dual
        edges = [
            (index_c[0], index_c[1])
            for index_c in must_link_c
            if index_c in constraints_dual_values and (
                constraints_dual_values[index_c] > 0.0)
            and (
                instance_model_dual[
                    my_mdl.instance_model.must_link_c[index_c]] > (
                    constraints_dual_values[index_c] + tol * (
                        constraints_dual_values[index_c]))
                or instance_model_dual[
                    my_mdl.instance_model.must_link_c[index_c]] > tol_value
            )
        ]

        # Add edges in batch
        conflict_graph.add_edges_from(edges)
    elif my_mdl.pb_number == 2:
        max_edges = len(my_mdl.instance_model.must_link_n)
        conflict_graph = nx.Graph(capacity=max_edges)

        # Store frequently used values
        must_link_n = my_mdl.instance_model.must_link_n
        instance_model_dual = my_mdl.instance_model.dual
        edges = [
            (index_c[0], index_c[1])
            for index_c in must_link_n
            if index_c in constraints_dual_values and (
                constraints_dual_values[index_c] > 0.0)
            and (
                instance_model_dual[
                    my_mdl.instance_model.must_link_n[index_c]] > (
                    constraints_dual_values[index_c] + tol * (
                        constraints_dual_values[index_c]))
                or instance_model_dual[
                    my_mdl.instance_model.must_link_n[index_c]] > tol_value
            )
        ]

        # Add edges in batch
        conflict_graph.add_edges_from(edges)
    return conflict_graph


def get_moving_containers_clust(
    my_mdl, constraints_dual_values, tol, tol_move, nb_containers, dict_id_c,
    df_clust, profiles
):
    """Get the list of moving containers from constraints dual values.

    :param my_mdl: _description_
    :type my_mdl: Model
    :param constraints_dual_values: _description_
    :type constraints_dual_values: Dict
    :param tol: _description_
    :type tol: float
    :param tol_move: _description_
    :type tol_move: float
    :param nb_containers: _description_
    :type nb_containers: int
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    :param df_clust: _description_
    :type df_clust: pd.DataFrame
    :param profiles: _description_
    :type profiles: np.array
    :return: _description_
    :rtype: Tuple[List, int, int, int, int]
    """
    mvg_containers = []
    conflict_graph = get_conflict_graph(my_mdl, constraints_dual_values, tol)

    graph_nodes = conflict_graph.number_of_nodes()
    graph_edges = conflict_graph.number_of_edges()
    list_indivs = sorted(
        conflict_graph.degree, key=lambda x: x[1], reverse=True)
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
        list_indivs = sorted(
            conflict_graph.degree, key=lambda x: x[1], reverse=True)

    return (mvg_containers, graph_nodes, graph_edges,
            max_deg, mean_deg)


# TODO to improve : very low dual values can change easily
# TODO choose container by most changing profile ?
def get_moving_containers_place(
    my_mdl, constraints_dual_values, tol, tol_move,
    nb_containers, working_df, dict_id_c
):
    """Get the list of moving containers from constraints dual values.

    :param my_mdl: _description_
    :type my_mdl: Model
    :param constraints_dual_values: _description_
    :type constraints_dual_values: Dict
    :param tol: _description_
    :type tol: float
    :param tol_move: _description_
    :type tol_move: float
    :param nb_containers: _description_
    :type nb_containers: int
    :param working_df: _description_
    :type working_df: pd.DataFrame
    :param dict_id_c: _description_
    :type dict_id_c: Dict
    :return: _description_
    :rtype: Tuple[List, int, int, int, int]
    """
    mvg_containers = []
    conflict_graph = get_conflict_graph(my_mdl, constraints_dual_values, tol)

    graph_nodes = conflict_graph.number_of_nodes()
    graph_edges = conflict_graph.number_of_edges()
    list_indivs = sorted(
        conflict_graph.degree, key=lambda x: x[1], reverse=True)
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
        list_indivs = sorted(
            conflict_graph.degree, key=lambda x: x[1], reverse=True)

    return (mvg_containers, graph_nodes, graph_edges,
            max_deg, mean_deg)


def get_container_tomove(c1, c2, working_df):
    """Get the container we want to move between c1 and c2.

    :param c1: _description_
    :type c1: int
    :param c2: _description_
    :type c2: int
    :param working_df: _description_
    :type working_df: pd.DataFrame
    :return: _description_
    :rtype: int
    """
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


def get_obj_value_host(t_min=None, t_max=None):
    """Get objectives value of current solution.

    :param t_min: _description_, defaults to None
    :type t_min: int, optional
    :param t_max: _description_, defaults to None
    :type t_max: int, optional
    :return: _description_
    :rtype: Tuple[int, float]
    """
    t_min = t_min or it.my_instance.df_host_evo[it.tick_field].min()
    t_max = t_max or it.my_instance.df_host_evo[it.tick_field].max()
    df_host = it.my_instance.df_host_evo[
        (it.my_instance.df_host_evo[it.tick_field] >= t_min)
        & (it.my_instance.df_host_evo[it.tick_field] <= t_max)]
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


def get_obj_value_indivs(df_indiv, t_min=None, t_max=None):
    """Get objective value of current solution (max delta).

    :param df_indiv: _description_
    :type df_indiv: pd.DataFrame
    :param t_min: _description_, defaults to None
    :type t_min: int, optional
    :param t_max: _description_, defaults to None
    :type t_max: int, optional
    :return: _description_
    :rtype: Tuple[int, float]
    """
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
