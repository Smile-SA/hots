"""HOTS optimization model definitions."""

from itertools import product
from typing import Any, Dict, List

from hots.plugins.clustering.builder import build_matrix_indiv_attr, build_similarity_matrix

import numpy as np

import pandas as pd

from pyomo import environ as pe
from pyomo.common.errors import ApplicationError
from pyomo.opt import TerminationCondition


class Model:
    """
    Encapsulates a HOTS optimization problem in Pyomo.
    All original methods (build_parameters, build_variables, etc.)
    become instance methods on this class.
    """

    def __init__(
        self,
        pb_number: int,
        df_indiv: pd.DataFrame,
        metric: str,
        dict_id_c: dict,
        df_meta: pd.DataFrame = None,
        nb_clusters: int = None,
        verbose: bool = False,
    ):
        """
        Initialize the optimization Model with data, parameters, and build the Pyomo instance.

        :param pb_number: Problem identifier (1 for clustering, 2 for placement).
        :param df_indiv: DataFrame of individual‑level usage data.
        :param metric: Name of the metric to optimize (e.g. 'cpu').
        :param dict_id_c: Mapping from container IDs to integer indices.
        :param df_meta: Host metadata DataFrame (optional; required for placement).
        :param nb_clusters: Number of clusters for clustering problems.
        :param verbose: If True, enables verbose debug output and writes LP files.
        """
        # --- Store inputs as attributes instead of using module‐globals (it.*) ---
        self.pb_number = pb_number
        self.df_indiv = df_indiv
        self.metric = metric
        self.dict_id_c = dict_id_c
        self.df_meta = df_meta
        self.nb_clusters = nb_clusters
        self.verbose = verbose

        # Extract from df_indiv for convenience
        self.tick_field = 'timestamp'  # or pass as extra arg if you need it parametric
        self.indiv_field = 'container_id'  # likewise if needed
        self.host_field = 'machine_id'
        self.metrics = [metric]

        # suppose you have numpy arrays u_prev, v_prev
        # model.u_matrix = u_prev
        # model.v_matrix = v_prev

        # --- Build the Pyomo model object ---
        self.mdl = pe.AbstractModel()
        self._build_model()

        # Build the data and create a concrete model
        self.create_data()

        if verbose:
            self.write_infile(fname=f'./debug_pb{self.pb_number}.lp')

        self.instance_model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    def _build_model(self):
        self.build_parameters()
        self.build_variables()
        self.build_constraints()
        self.build_objective()

    def build_parameters(self):
        """
        Build all Pyomo Params and Sets by computing:
          - similarity matrix w
          - variance/distance matrix dv
          - clustering adjacency u (defaults to zeros)
          - placement adjacency v (defaults to zeros)
        from self.df_indiv and any injected self.u_matrix/self.v_matrix.
        """
        # 1) Build the container×tick matrix and similarity/variance
        mat = build_matrix_indiv_attr(
            self.df_indiv,
            self.tick_field,
            self.indiv_field,
            self.metrics,
            self.dict_id_c,
        )
        n = mat.shape[0]
        w = build_similarity_matrix(mat)
        dv = build_similarity_matrix(mat)  # or your own variance func

        # 2) Prepare list of container IDs in correct order
        #    dict_id_c maps ID → integer index 0..n-1
        id_list = [None] * n
        for cid, idx in self.dict_id_c.items():
            id_list[idx] = cid

        # 3) Retrieve prior-solution adjacency or zeros
        u = getattr(self, 'u_matrix', np.zeros((n, n), dtype=int))
        v = getattr(self, 'v_matrix', np.zeros((n, n), dtype=int))

        # --- Define Sets & Params on container IDs ---
        self.mdl.C = pe.Set(initialize=id_list, doc='Container IDs')
        self.mdl.c = pe.Param(within=pe.NonNegativeIntegers, mutable=True)

        # sol_u: clustering adjacency param indexed by (container_j, container_i)
        sol_u_dict = {
            (id_list[j], id_list[i]): int(u[i][j])
            for i, j in product(range(n), range(n))
        }
        self.mdl.sol_u = pe.Param(
            self.mdl.C, self.mdl.C, initialize=sol_u_dict, mutable=True
        )

        # sol_v: placement adjacency
        sol_v_dict = {
            (id_list[j], id_list[i]): int(v[i][j])
            for i, j in product(range(n), range(n))
        }
        self.mdl.sol_v = pe.Param(
            self.mdl.C, self.mdl.C, initialize=sol_v_dict, mutable=True
        )

        # --- Problem‑specific parameters ---
        if self.pb_number == 1:
            # Clustering: set of clusters K and similarity w
            self.mdl.K = pe.Set(initialize=list(range(self.nb_clusters)))
            self.mdl.k = pe.Param(within=pe.NonNegativeIntegers)
            w_dict = {
                (id_list[j], id_list[i]): float(w[i][j])
                for i, j in product(range(n), range(n))
            }
            self.mdl.w = pe.Param(
                self.mdl.C, self.mdl.C, initialize=w_dict, mutable=True
            )

        elif self.pb_number == 2:
            # Placement: nodes N, ticks T, capacities, consumption, and dv
            # (assumes create_data has already loaded cap, cons, etc.)
            self.mdl.N = pe.Set(initialize=self.data[None]['N'])
            self.mdl.n = pe.Param(within=pe.NonNegativeIntegers)
            self.mdl.cap = pe.Param(self.mdl.N)

            self.mdl.T = pe.Set(initialize=self.data[None]['T'])
            self.mdl.t = pe.Param(within=pe.NonNegativeIntegers)
            self.mdl.Ccons = pe.Set(initialize=self.data[None]['Ccons'])
            self.mdl.cons = pe.Param(
                self.mdl.Ccons,
                self.mdl.T,
                initialize=self.data[None]['cons'],
                mutable=True,
            )

            dv_dict = {
                (id_list[j], id_list[i]): float(dv[i][j])
                for i, j in product(range(n), range(n))
            }
            self.mdl.dv = pe.Param(
                self.mdl.C, self.mdl.C, initialize=dv_dict, mutable=True
            )

        else:
            raise ValueError(f'Unsupported pb_number: {self.pb_number}')

    def build_variables(self):
        """
        Define decision variables for clustering (pb_number=1)
        or placement (pb_number=2).
        """
        if self.pb_number == 1:
            # --- Clustering variables ---
            # yc[k] = 1 if cluster k is opened
            self.mdl.yc = pe.Var(
                self.mdl.K, domain=pe.Binary, doc='Cluster-open indicator'
            )
            # assign[c,k] = 1 if container c is assigned to cluster k
            self.mdl.assign = pe.Var(
                self.mdl.C,
                self.mdl.K,
                domain=pe.Binary,
                doc='Container-to-cluster assignment',
            )
            # dist[i,j] = distance between containers i and j
            self.mdl.dist = pe.Var(
                self.mdl.C,
                self.mdl.C,
                domain=pe.NonNegativeReals,
                doc='Inter-container distance',
            )
            # u[c1,c2] = 1 if containers c1 and c2 are assigned to the same cluster
            self.mdl.u = pe.Var(
                self.mdl.C,
                self.mdl.C,
                domain=pe.Binary,
                doc='Pairwise cluster adjacency',
            )

        elif self.pb_number == 2:
            # --- Placement variables ---
            self.mdl.place = pe.Var(
                self.mdl.C,
                self.mdl.N,
                self.mdl.T,
                domain=pe.Binary,
                doc='Container-to-node-time placement',
            )
            self.mdl.load = pe.Var(
                self.mdl.N,
                self.mdl.T,
                domain=pe.NonNegativeReals,
                doc='Node load over time',
            )

        else:
            raise ValueError(f'Unsupported pb_number: {self.pb_number}')

    def build_constraints(self):
        """
        Build all the constraints for clustering (pb_number=1)
        or placement (pb_number=2).
        """
        if self.pb_number == 1:
            # (1) each container assigned to exactly one cluster
            self.mdl.clust_assign = pe.Constraint(
                self.mdl.C, rule=self._clust_assign_rule
            )
            # (2) you can only open a cluster if at least one container is in it
            self.mdl.open_cluster = pe.Constraint(
                self.mdl.C, self.mdl.K, rule=self._open_cluster_rule
            )
            # (3) limit on total open clusters
            self.mdl.max_clusters = pe.Constraint(rule=self._max_clusters_rule)
            self.mdl.link_u1 = pe.Constraint(
                self.mdl.C, self.mdl.C, self.mdl.K, rule=self._link_u1_rule
            )
            self.mdl.link_u2 = pe.Constraint(
                self.mdl.C, self.mdl.C, self.mdl.K, rule=self._link_u2_rule
            )
            self.mdl.link_u3 = pe.Constraint(
                self.mdl.C, self.mdl.C, self.mdl.K, rule=self._link_u3_rule
            )

        elif self.pb_number == 2:
            # (1) capacity constraint: load on each node ≤ cap
            self.mdl.capacity = pe.Constraint(
                self.mdl.N, self.mdl.T, rule=self._capacity_rule
            )
            # (2) flow conservation: consumption matches load
            self.mdl.flow_conservation = pe.Constraint(
                self.mdl.Ccons, self.mdl.T, rule=self._flow_conservation_rule
            )
        else:
            raise ValueError(f'Unsupported pb_number: {self.pb_number}')

    def add_mustlink(self):
        """Add mustLink constraints for fixing solution."""
        if self.pb_number == 1:
            self.mdl.must_link_c = pe.Constraint(
                self.mdl.C, self.mdl.C, rule=self.must_link_c_
            )
        if self.pb_number == 2:
            self.mdl.must_link_n = pe.Constraint(
                self.mdl.C, self.mdl.C, rule=self.must_link_n_
            )

    def add_mustlink_instance(self):
        """Add mustLink constraints for fixing solution."""
        if self.pb_number == 1:
            self.instance_model.must_link_c = pe.Constraint(
                self.instance_model.C, self.instance_model.C, rule=self.must_link_c_
            )
        if self.pb_number == 2:
            self.instance_model.must_link_n = pe.Constraint(
                self.instance_model.C, self.instance_model.C, rule=self.must_link_n_
            )

    def build_objective(self):
        """Build the objective."""
        if self.pb_number == 1:
            self.mdl.obj = pe.Objective(rule=min_dissim_, sense=pe.minimize)
        elif self.pb_number == 2:
            self.mdl.obj = pe.Objective(rule=min_coloc_cluster_, sense=pe.minimize)

    def create_data(self):
        """
        Build the data dictionary for instantiating the abstract model,
        then call create_instance() to get a concrete model ready to solve.
        """
        # Shortcuts
        df_indiv = self.df_indiv
        df_meta = self.df_meta or pd.DataFrame()
        metric = self.metric
        id_map = self.dict_id_c
        nf = self.indiv_field
        hf = self.host_field
        tf = self.tick_field

        if self.pb_number == 1:
            # Clustering formulation
            self.data = {
                None: {
                    'c': {None: df_indiv[nf].nunique()},
                    'C': {None: list(id_map.keys())},
                    'k': {None: self.nb_clusters},
                    'K': {None: list(range(self.nb_clusters))},
                }
            }

        elif self.pb_number == 2:
            # Placement formulation

            # 1) node capacities
            cap = {}
            for node, grp in df_meta.groupby(hf):
                cap[node] = float(grp[metric].iloc[0])

            # 2) container × tick consumptions
            cons = {}
            for (cid, tick), grp in df_indiv.groupby([nf, tf]):
                cons[(cid, tick)] = float(grp[metric].iloc[0])

            self.data = {
                None: {
                    'n': {None: df_meta[hf].nunique()},
                    'c': {None: df_indiv[nf].nunique()},
                    't': {None: df_indiv[tf].nunique()},
                    'N': {None: df_meta[hf].unique().tolist()},
                    'C': {None: list(id_map.keys())},
                    'Ccons': {None: df_indiv[nf].unique().tolist()},
                    'cap': cap,
                    'T': {None: df_indiv[tf].unique().tolist()},
                    'cons': cons,
                }
            }
        else:
            raise ValueError(f'Unsupported pb_number: {self.pb_number!r}')

        # Finally, instantiate the model with data:
        self.instance_model = self.mdl.create_instance(self.data)

    def conso_n_t(self, mdl, node, t):
        """Express the total consumption of node at time t.

        :param mdl: Pyomo model object
        :type mdl: pe.AbstractModel
        :param node: Node index
        :type node: int
        :param t: Time index
        :type t: int
        :return: Total consumption of the node at time t
        :rtype: float
        """
        return sum(
            mdl.x[cont, node] * self.cons[cont_c][t]
            for cont, cont_c in zip(mdl.C, mdl.Ccons)
        )

    def mean(self, mdl, node):
        """Compute the mean consumption on a node across all ticks."""
        return sum(mdl.load[node, t] for t in mdl.T) / len(mdl.T)

    def write_infile(self, fname=None):
        """Dump the concrete model to an LP file for debugging."""
        if fname is None:
            fname = './py_clustering.lp' if self.pb_number == 1 else './py_placement.lp'
        self.instance_model.write(fname, io_options={'symbolic_solver_labels': True})

    def solve(self, solver: str = 'glpk', verbose: bool = False):
        """
        Solve the concrete instance_model with the given solver.

        :param solver: Name of the Pyomo solver to use (e.g. 'glpk').
        :param verbose: If True, prints constraint checks, status, and objective.
        :return: self, with instance_model solved and suffixes/duals populated.
        """
        opt = pe.SolverFactory(solver)
        results = self._attempt_solve(opt, verbose)

        if verbose:
            # Check each constraint for violations
            self._check_constraints()
            # Print solver termination status
            self._print_solver_status(results)
            # Print the final objective value
            self._print_objective_value()

        return self

    def _attempt_solve(self, opt, verbose):
        """Attempt to solve the model and handles solver errors.

        :param opt: Pyomo solver object.
        :type opt: pe.SolverFactory
        :param verbose: Enable / disable logs during solve process.
        :type verbose: bool
        """
        try:
            return opt.solve(self.instance_model, tee=verbose)
        except ApplicationError as e:
            print(f'Solver error: {e}')
            return None

    def _check_constraints(self):
        """Check constraint violations in the model."""
        for c in self.instance_model.component_objects(pe.Constraint, active=True):
            print(f'\n🔍 Checking Constraint: {c.name}')
            for index in c:
                self._check_constraint_violation(c, index)

    def _check_constraint_violation(self, constraint, index):
        """Check if a specific constraint is violated.

        :param constraint: Pyomo constraint object
        :type constraint: pe.Constraint
        :param index: Index of the constraint
        :type index: int
        """
        lhs = constraint[index].body()
        upper, lower = constraint[index].upper, constraint[index].lower

        if upper is not None and lhs > upper:
            print(f'  ❌ Constraint {constraint.name}[{index}] is violated!')
            print(f'     Expected: LHS ≤ {upper}, Got: LHS = {lhs} (TOO HIGH!)')
        elif lower is not None and lhs < lower:
            print(f'  ❌ Constraint {constraint.name}[{index}] is violated!')
            print(f'     Expected: LHS ≥ {lower}, Got: LHS = {lhs} (TOO LOW!)')
        else:
            print(f'  ✅ Constraint {constraint.name}[{index}] holds. (LHS = {lhs})')

    def _print_solver_status(self, results):
        """Print solver termination condition.

        :param results: Results object returned by the solver
        :type results: SolverResults
        """
        print('--------------------------------------------------')
        condition = results.solver.termination_condition if results else None

        status_messages = {
            TerminationCondition.infeasible: 'Problem is infeasible!',
            TerminationCondition.unbounded: 'Problem is unbounded!',
            TerminationCondition.optimal: 'Solution found.',
        }
        print(status_messages.get(condition, f'Solver status: {condition}'))
        print('--------------------------------------------------')

    def _print_objective_value(self):
        """Print the objective function value."""
        print(f'Objective function value: {pe.value(self.instance_model.obj)}')

    # TODO generalize with others constraints than mustlink
    def update_adjacency_clust_constraints(self, u):
        """Update constraints fixing u variables from new adjacency matrix.

        :param u: Clustering adjacency matrix
        :type u: np.array
        """
        self.instance_model.del_component(self.instance_model.must_link_c)
        self.update_sol_u(u)
        self.add_mustlink_instance()

    def update_sol_u(self, u):
        """Update directly the sol_u param in instance from new u matrix.

        :param u: Clustering adjacency matrix
        :type u: np.array
        """
        for i, j in product(range(len(u)), range(len(u[0]))):
            self.instance_model.sol_u[(i, j)] = u[i][j]

    # TODO generalize with others constraints than mustlink
    def update_adjacency_place_constraints(self, v_matrix):
        """
        After you compute a new placement adjacency v_matrix (np.array),
        update the existing instance_model’s must_link constraints.
        """
        # 1) delete the old constraint
        self.instance_model.del_component('must_link_n')

        # 2) update the sol_v param in the _instance_, using ID keys
        n = len(self.dict_id_c)
        id_list = [None] * n
        for cid, idx in self.dict_id_c.items():
            id_list[idx] = cid

        for i, j in product(range(n), range(n)):
            key = (id_list[j], id_list[i])
            self.instance_model.sol_v[key] = int(v_matrix[i][j])

        # 3) rebuild the must_link constraint under the new data
        self.add_mustlink_instance()

    def update_sol_v(self, v_matrix):
        """Directly overwrite the concrete model’s sol_v param values."""
        n = len(self.dict_id_c)
        id_list = [None] * n
        for cid, idx in self.dict_id_c.items():
            id_list[idx] = cid

        for i, j in product(range(n), range(n)):
            key = (id_list[j], id_list[i])
            self.instance_model.sol_v[key] = int(v_matrix[i][j])

    def update_obj_clustering(self, w_matrix):
        """
        After computing a new similarity matrix w_matrix (np.array),
        update the objective’s coefficients and rebuild the suffixes.
        """
        from pyomo import environ as pe

        # Update the w param in the ABSTRACT instance_model
        n = len(self.dict_id_c)
        id_list = [None] * n
        for cid, idx in self.dict_id_c.items():
            id_list[idx] = cid

        # 1) overwrite each entry of w in the concrete model
        for i, j in product(range(n), range(n)):
            key = (id_list[j], id_list[i])
            self.instance_model.w[key] = float(w_matrix[i][j])

        # 2) rebuild the objective (if it’s data‐dependent)
        #    [assuming you have an `obj_rule` defined]
        self.instance_model.del_component('obj')
        self.instance_model.obj = pe.Objective(rule=self.obj_rule, sense=pe.maximize)

        # 3) re‐fetch duals
        self.instance_model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    def update_w(self, w):
        """Update directly the w param in instance from new w matrix.

        :param w: Similarity matrix
        :type w: np.array
        """
        for i, j in product(range(len(w)), range(len(w[0]))):
            self.instance_model.w[(i, j)] = w[i][j]

    def update_obj_place(self, dv):
        """Update the objective for placement with new dv matrix.

        :param dv: Clustering variance matrix
        :type dv: np.array
        """
        self.update_dv(dv)
        self.instance_model.obj = sum(
            [
                self.instance_model.sol_u[(i, j)] * (self.instance_model.v[(i, j)])
                for i, j in product(self.instance_model.C, self.instance_model.C)
                if i < j
            ]
        ) + sum(
            [
                (1 - self.instance_model.sol_u[(i, j)])
                * (self.instance_model.v[(i, j)] * self.instance_model.dv[(i, j)])
                for i, j in product(self.instance_model.C, self.instance_model.C)
                if i < j
            ]
        )

    def update_dv(self, dv):
        """Update directly the dv param in instance from new dv matrix.

        :param dv: Clustering variance matrix
        :type dv: np.array
        """
        for i, j in product(range(len(dv)), range(len(dv[0]))):
            self.instance_model.dv[(i, j)] = dv[i][j]

    def update_size_model(self, new_df_indiv=None, verbose=False):
        """
        Rebuild the abstract & concrete models when the set of containers
        changes (e.g. new/deleted containers).

        :param new_df_indiv: optional updated individual‐level data
        :type new_df_indiv: pd.DataFrame
        :param verbose: if True, write LP file before solving
        :type verbose: bool
        """
        # 1) Update the raw data if provided
        if new_df_indiv is not None:
            self.df_indiv = new_df_indiv

        # 2) Reset the abstract model
        self.mdl = pe.AbstractModel()

        # 3) Re‑build the entire model (sets, params, vars, constraints, objective)
        #    This uses your unified setup in __init__ (via _build_model)
        self._build_model()

        # 4) (Re‑)create the concrete instance with the latest data
        self.create_data()

        # 5) Optionally dump the LP for inspection
        if verbose:
            self.write_infile(fname=f'./debug_pb{self.pb_number}.lp')

        # 6) Re‑attach dual suffix for post‑solve inspection
        self.instance_model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    def fill_dual_values(self) -> Dict[Any, float]:
        """
        Extract duals for the 'must_link' constraints after solve().
        Returns a mapping from index (container or node tuple) to its dual value.
        """
        dual = self.instance_model.dual
        # pick the right constraint component
        if self.pb_number == 1:
            con = getattr(self.instance_model, 'must_link_c', None)
        else:
            con = getattr(self.instance_model, 'must_link_n', None)

        if con is None:
            return {}

        # build the mapping
        return {
            idx: dual[con[idx]]
            for idx in con  # iterates over all index tuples in the Constraint
        }

    def _clust_assign_rule(self, mdl, c):
        # sum of assign[c,k] over clusters k must equal 1
        return sum(mdl.assign[c, k] for k in mdl.K) == 1

    def _open_cluster_rule(self, mdl, c, k):
        # you can only assign container c to cluster k
        # if that cluster has been ‘opened’ via yc[k] = 1
        return mdl.assign[c, k] <= mdl.yc[k]

    def _max_clusters_rule(self, mdl):
        # total number of opened clusters ≤ k
        return sum(mdl.yc[k] for k in mdl.K) <= mdl.k

    def _link_u1_rule(self, mdl, i, j, k):
        # u[i,j] ≥ assign[i,k] + assign[j,k] – 1
        return mdl.u[i, j] >= mdl.assign[i, k] + mdl.assign[j, k] - 1

    def _link_u2_rule(self, mdl, i, j, k):
        # u[i,j] ≤ assign[i,k]
        return mdl.u[i, j] <= mdl.assign[i, k]

    def _link_u3_rule(self, mdl, i, j, k):
        # u[i,j] ≤ assign[j,k]
        return mdl.u[i, j] <= mdl.assign[j, k]

    def _capacity_rule(self, mdl, node, t):
        # load[node,t] = sum(place[c,node,t] * cons[c,t])
        return mdl.load[node, t] <= mdl.cap[node]

    def _flow_conservation_rule(self, mdl, c, t):
        # sum of place[c,node,t] over nodes = 1 (each container runs somewhere)
        return sum(mdl.place[c, n, t] for n in mdl.N) == 1

    def open_node_(self, mdl, container, node):
        """Express the opening node constraint.

        :param mdl: Pyomo model object
        :type mdl: pe.AbstractModel
        :param container: Container index
        :type container: int
        :param node: Node index
        :type node: int
        :return: Expression for opening node constraint
        :rtype: Expression
        """
        return mdl.x[container, node] <= mdl.a[node]

    def assignment_(self, mdl, container):
        """Express the assignment constraint.

        :param mdl: Pyomo model object
        :type mdl: pe.AbstractModel
        :param container: Container index
        :type container: int
        :return: Expression for container assignment
        :rtype: Expression
        """
        return sum(mdl.x[container, node] for node in mdl.N) == 1

    def open_nodes_(self, mdl):
        """Express the numbers of open nodes.

        :param mdl: Pyomo model object
        :type mdl: pe.AbstractModel
        :return: Number of open nodes
        :rtype: int
        """
        return sum(mdl.a[m] for m in mdl.N)

    def must_link_c_(self, mdl, i, j):
        """Express the clustering mustlink constraint.

        :param mdl: Pyomo model object
        :type mdl: pe.AbstractModel
        :param i: Container i index
        :type i: int
        :param j: Container j index
        :type j: int
        :return: Expression for must link constraint
        :rtype: Expression
        """
        uu = mdl.sol_u[(i, j)].value
        if uu == 1:
            return mdl.u[(i, j)] == 1
        else:
            return pe.Constraint.Skip

    def must_link_n_(self, mdl, i, j):
        """Express the placement mustlink constraint.

        :param mdl: Pyomo model object
        :type mdl: pe.AbstractModel
        :param i: Container i index
        :type i: int
        :param j: Container j index
        :type j: int
        :return: Expression for must link constraint
        :rtype: Expression
        """
        vv = mdl.sol_v[(i, j)].value
        if vv == 1:
            return mdl.v[(i, j)] == 1
        else:
            return pe.Constraint.Skip

    def extract_moves(self) -> List[Dict[str, Any]]:
        """
        Return the placement moves encoded in the solved instance_model.
        Each entry is a dict with:
          - container_id: the container key
          - node: the target node key
          - tick: the time tick
        """
        moves: List[Dict[str, Any]] = []
        inst = self.instance_model

        # only for placement problems
        if self.pb_number == 2:
            for c in inst.C:
                for t in inst.T:
                    for n in inst.N:
                        # check the binary variable place[c,n,t]
                        if pe.value(inst.place[c, n, t]) > 0.5:
                            moves.append({'container_id': c, 'node': n, 'tick': t})
        return moves


def min_dissim_(mdl):
    """Express the within clusters dissimilarities.

    :param mdl: Pyomo model object
    :type mdl: pe.AbstractModel
    :return: Within clusters dissimilarities
    :rtype: float
    """
    return sum(
        [mdl.u[(i, j)] * mdl.w[(i, j)] for i, j in product(mdl.C, mdl.C) if i < j]
    )


def min_coloc_cluster_(mdl):
    """Express the placement minimization objective from clustering.

    :param mdl: Pyomo model object
    :type mdl: pe.AbstractModel
    :return: Objective for placement optimization
    :rtype: Expression
    """
    return sum(
        [mdl.sol_u[(i, j)] * mdl.v[(i, j)] for i, j in product(mdl.C, mdl.C) if i < j]
    ) + sum(
        [
            ((1 - mdl.sol_u[(i, j)]) * mdl.v[(i, j)] * mdl.dv[(i, j)])
            for i, j in product(mdl.C, mdl.C)
            if i < j
        ]
    )
