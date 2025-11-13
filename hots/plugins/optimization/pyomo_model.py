"""Pyomo-based optimization plugin for HOTS (single concrete implementation)."""

from itertools import product
from typing import Any, Dict, Optional

from hots.core.interfaces import OptimizationPlugin

import numpy as np

from pyomo import environ as pe
from pyomo.common.errors import ApplicationError
from pyomo.common.numeric_types import value as pyo_value
from pyomo.opt import TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints


class PyomoModel(OptimizationPlugin):
    """Concrete optimization backend using Pyomo."""

    # ---------- lifecycle ----------
    def __init__(self, params: dict, instance):
        """Initialize the optimization model."""
        self.instance = instance
        self.pb_number = params.get('pb_number', 1)
        self.solver = params.get('solver', 'glpk')
        self.verbose = params.get('verbose', False)
        self.last_duals = None

        # derived shortcuts from instance/config
        self.df_indiv = instance.df_indiv
        self.df_meta = getattr(instance, 'df_meta', None)
        self.metric = instance.config.metrics[0]
        self.nb_clusters = getattr(instance.config.clustering, 'nb_clusters', None)
        self.dict_id_c = instance.get_id_map()

        # schema (can be made configurable later)
        self.tick_field = getattr(instance.config, 'tick_field', 'timestamp')
        self.indiv_field = getattr(instance.config, 'individual_field', 'container_id')
        self.host_field = getattr(instance.config, 'host_field', 'machine_id')

        # matrices (passed at build time)
        self.u_mat = None
        self.w_mat = None
        self.v_mat = None
        self.dv_mat = None

        # pyomo objects
        self.mdl = None
        self.instance_model = None
        self.id_list = None
        self.data = None

    # -- OptimizationPlugin API --
    def build(self, *, u_mat=None, w_mat=None, v_mat=None, dv_mat=None):
        """Create and store a concrete Pyomo model instance for this pb_number."""
        # hold matrices
        self.u_mat = u_mat
        if self.pb_number == 1:
            self.w_mat = w_mat
        elif self.pb_number == 2:
            self.v_mat = v_mat
            self.dv_mat = dv_mat

        # abstract model
        self.mdl = pe.AbstractModel()

        # build sets/params/vars/cons/obj
        self._build_model()

        # instantiate with data
        self.create_data()
        if self.verbose:
            self.write_infile(fname=f'./debug_pb{self.pb_number}.lp')

        # enable duals
        self.instance_model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    def solve(self, *, solver: Optional[str] = None):
        """Solve the current concrete instance (labels optional for compatibility)."""
        opt = pe.SolverFactory(solver or self.solver)
        results = self._attempt_solve(opt)

        if self.verbose:
            self._print_solver_status(results)
            tc = results.solver.termination_condition if results else None
            if tc in (TerminationCondition.optimal, TerminationCondition.feasible):
                self._print_objective_value()
                self._check_constraints(evaluate=True)
            else:
                print('\nModel not solved to a feasible status; listing infeasible constraints:')
                log_infeasible_constraints(self.instance_model, log_expression=True, tol=1e-6)

    # ---------- model build ----------
    def _build_model(self):
        self.build_parameters()
        self.build_variables()
        self.build_constraints()
        self.add_mustlink()
        self.build_objective()

    def build_parameters(self):
        """Build all Pyomo Params and Sets from data."""
        n = self.u_mat.shape[0]

        # 2) Prepare list of container IDs in correct order
        #    dict_id_c maps ID → integer index 0..n-1
        self.id_list = [None] * n
        for cid, idx in self.dict_id_c.items():
            self.id_list[idx] = cid

        # 3) Retrieve prior-solution adjacency or zeros
        u = getattr(self, 'u_mat', np.zeros((n, n), dtype=int))
        v = getattr(self, 'v_mat', np.zeros((n, n), dtype=int))

        # --- Define Sets & Params on container IDs ---
        self.mdl.C = pe.Set(initialize=self.id_list, doc='Container IDs')
        self.mdl.c = pe.Param(within=pe.NonNegativeIntegers, mutable=True)

        # sol_u: clustering adjacency param indexed by (container_j, container_i)
        sol_u_dict = {
            (self.id_list[j], self.id_list[i]): int(u[i][j])
            for i, j in product(range(n), range(n))
        }
        self.mdl.sol_u = pe.Param(
            self.mdl.C, self.mdl.C, initialize=sol_u_dict, mutable=True
        )

        # --- Problem‑specific parameters ---
        if self.pb_number == 1:
            # Clustering: set of clusters K and similarity w
            self.mdl.K = pe.Set(initialize=list(range(self.nb_clusters)))
            self.mdl.k = pe.Param(within=pe.NonNegativeIntegers)
            w_dict = {
                (self.id_list[j], self.id_list[i]): float(self.w_mat[i][j])
                for i, j in product(range(n), range(n))
            }
            self.mdl.w = pe.Param(
                self.mdl.C, self.mdl.C, initialize=w_dict, mutable=True
            )

        elif self.pb_number == 2:
            # Placement: nodes N, ticks T, capacities, consumption, and dv
            # (assumes create_data has already loaded cap, cons, etc.)
            self.mdl.N = pe.Set(dimen=1)
            self.mdl.n = pe.Param(within=pe.NonNegativeIntegers)
            self.mdl.cap = pe.Param(self.mdl.N)

            self.mdl.T = pe.Set(dimen=1)
            self.mdl.t = pe.Param(within=pe.NonNegativeIntegers)
            self.mdl.Ccons = pe.Set(dimen=1)
            self.mdl.cons = pe.Param(
                self.mdl.Ccons,
                self.mdl.T,
                mutable=True,
            )
            dv_dict = {
                (self.id_list[j], self.id_list[i]): float(self.dv_mat[i][j])
                for i, j in product(range(n), range(n))
            }
            self.mdl.dv = pe.Param(
                self.mdl.C, self.mdl.C, initialize=dv_dict, mutable=True
            )
            # sol_v: placement adjacency
            sol_v_dict = {
                (self.id_list[j], self.id_list[i]): int(v[i][j])
                for i, j in product(range(n), range(n))
            }
            self.mdl.sol_v = pe.Param(
                self.mdl.C, self.mdl.C, initialize=sol_v_dict, mutable=True
            )

        else:
            raise ValueError(f'Unsupported pb_number: {self.pb_number}')

    def build_variables(self):
        """Define decision variables for clustering (1) or business problem (2)."""
        if self.pb_number == 1:
            # --- Clustering variables ---
            # yc[k] = 1 if cluster k is opened
            self.mdl.yc = pe.Var(
                self.mdl.K,
                domain=pe.NonNegativeReals,
                bounds=(0, 1),
                initialize=0,
                doc='Cluster-open indicator'
            )
            # assign[c,k] = 1 if container c is assigned to cluster k
            self.mdl.assign = pe.Var(
                self.mdl.C,
                self.mdl.K,
                domain=pe.NonNegativeReals,
                bounds=(0, 1),
                initialize=0,
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
                domain=pe.NonNegativeReals,
                bounds=(0, 1),
                initialize=0,
                doc='Pairwise cluster adjacency',
            )

        elif self.pb_number == 2:
            # --- Placement variables ---
            # place(i, n) = 1 if container i is placed on node n
            self.mdl.place = pe.Var(
                self.mdl.C,
                self.mdl.N,
                domain=pe.NonNegativeReals,
                bounds=(0, 1),
                initialize=0,
                doc='Container-to-node placement',
            )
            # a(n) = 1 if node n is used
            self.mdl.a = pe.Var(
                self.mdl.N,
                domain=pe.NonNegativeReals,
                bounds=(0, 1),
                initialize=0,
                doc='Node-open indicator'
            )
            # v(i,j) = 1 if containers i and j are placed on the same node
            self.mdl.v = pe.Var(
                self.mdl.C,
                self.mdl.C,
                domain=pe.NonNegativeReals,
                bounds=(0, 1),
                initialize=0,
                doc='Pairwise node adjacency'
            )
            self.mdl.node_load = pe.Var(
                self.mdl.N,
                self.mdl.T,
                domain=pe.NonNegativeReals,
                doc='Node load over time',
            )

        else:
            raise ValueError(f'Unsupported pb_number: {self.pb_number}')

    def build_constraints(self):
        """Build all the constraints."""
        if self.pb_number == 1:
            # (1) each container assigned to exactly one cluster
            self.mdl.clust_assign = pe.Constraint(
                self.mdl.C, rule=_clust_assign_rule
            )
            # (2) you can only open a cluster if at least one container is in it
            self.mdl.open_cluster = pe.Constraint(
                self.mdl.C, self.mdl.K, rule=_open_cluster_rule
            )
            # (3) limit on total open clusters
            self.mdl.max_clusters = pe.Constraint(rule=_max_clusters_rule)
            self.mdl.link_u1 = pe.Constraint(
                self.mdl.C, self.mdl.C, self.mdl.K, rule=_link_u1_rule
            )
            self.mdl.link_u2 = pe.Constraint(
                self.mdl.C, self.mdl.C, rule=_link_u2_rule
            )
            self.mdl.link_u3 = pe.Constraint(
                self.mdl.C, self.mdl.C, rule=_link_u3_rule
            )

        elif self.pb_number == 2:
            # (1) capacity constraint: load on each node ≤ cap
            self.mdl.capacity = pe.Constraint(
                self.mdl.N, self.mdl.T, rule=_capacity_rule
            )
            # (2) open node if at least one container is in it
            self.mdl.open_node = pe.Constraint(
                self.mdl.C, self.mdl.N, rule=_open_node
            )
            # (3) each container placed to exactly one node
            self.mdl.assignment = pe.Constraint(
                self.mdl.C, rule=_assignment
            )
            self.mdl.link_v1 = pe.Constraint(
                self.mdl.C, self.mdl.C, self.mdl.N, rule=_link_v1_rule
            )
            self.mdl.link_v2 = pe.Constraint(
                self.mdl.C, self.mdl.C, rule=_link_v2_rule
            )
            self.mdl.link_v3 = pe.Constraint(
                self.mdl.C, self.mdl.C, rule=_link_v3_rule
            )
            # (3) flow conservation: consumption matches load
            # self.mdl.flow_conservation = pe.Constraint(
            #     self.mdl.Ccons, self.mdl.T, rule=_flow_conservation_rule
            # )
        else:
            raise ValueError(f'Unsupported pb_number: {self.pb_number}')

    def add_mustlink(self):
        """Add mustLink constraints for fixing solution."""
        if self.pb_number == 1:
            self.mdl.must_link_c = pe.Constraint(self.mdl.C, self.mdl.C, rule=_must_link_c)
        if self.pb_number == 2:
            self.mdl.must_link_n = pe.Constraint(self.mdl.C, self.mdl.C, rule=_must_link_n)

    def build_objective(self):
        """Define objective function."""
        if self.pb_number == 1:
            self.mdl.obj = pe.Objective(rule=_min_dissim, sense=pe.minimize)
        elif self.pb_number == 2:
            self.mdl.obj = pe.Objective(rule=_min_coloc_cluster, sense=pe.minimize)

    def create_data(self):
        """Build data dictionnary to instanciate abstract model and build ready-to-solve model."""
        # Shortcuts
        df_indiv = self.df_indiv
        df_meta = self.df_meta
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

    # ---------- helpers & rules (migrated) ----------
    def write_infile(self, fname=None):
        """Export optimization model in file."""
        if fname is None:
            fname = './py_clustering.lp' if self.pb_number == 1 else './py_placement.lp'
        self.instance_model.write(fname, io_options={'symbolic_solver_labels': True})

    def _attempt_solve(self, opt):
        try:
            return opt.solve(self.instance_model, tee=self.verbose)
        except ApplicationError as e:
            print(f'Solver error: {e}')
            return None

    def _check_constraints(self, evaluate: bool = True):
        for c in self.instance_model.component_objects(pe.Constraint, active=True):
            for index in c:
                self._check_constraint_violation(c, index, evaluate=evaluate)

    def _check_constraint_violation(self, constraint, index, evaluate: bool):
        con = constraint[index]
        expr = con.body
        if not evaluate:
            print(
                f'(symbolic) {constraint.name}[{index}]: {expr} <= {con.upper}  and >= {con.lower}'
            )
            return
        lhs = pyo_value(expr, exception=False)
        if lhs is None:
            return
        upper, lower = con.upper, con.lower
        if upper is not None and lhs > upper + 1e-8:
            print(f'  ❌ {constraint.name}[{index}] violated: LHS={lhs} > UB={upper}')
        elif lower is not None and lhs < lower - 1e-8:
            print(f'  ❌ {constraint.name}[{index}] violated: LHS={lhs} < LB={lower}')

    def _print_solver_status(self, results):
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
        print(f'Objective function value: {pe.value(self.instance_model.obj)}')

    # updates / mustlink (kept)
    def add_mustlink_instance(self):
        """Add mustLink constraints for fixing solution."""
        if self.pb_number == 1:
            self.instance_model.must_link_c = pe.Constraint(
                self.instance_model.C, self.instance_model.C, rule=_must_link_c
            )
        if self.pb_number == 2:
            self.instance_model.must_link_n = pe.Constraint(
                self.instance_model.C, self.instance_model.C, rule=_must_link_n
            )

    def update_adjacency_clust_constraints(self, u):
        """Update constraints fixing u variables from new adjacency matrix."""
        self.instance_model.del_component(self.instance_model.must_link_c)
        self.update_sol_u(u)
        self.add_mustlink_instance()

    def update_sol_u(self, u):
        """Update directly u variable from new adjacency matrix."""
        n = len(self.id_list)
        for i in range(n):
            for j in range(n):
                cj = self.id_list[j]
                ci = self.id_list[i]
                self.instance_model.sol_u[(cj, ci)] = int(u[i][j])

    def update_adjacency_place_constraints(self, v_matrix):
        """Update constraints fixing v variables from new adjacency matrix."""
        self.instance_model.del_component('must_link_n')
        n = len(self.dict_id_c)
        for i, j in product(range(n), range(n)):
            key = (self.id_list[j], self.id_list[i])
            self.instance_model.sol_v[key] = int(v_matrix[i][j])
        self.add_mustlink_instance()

    def update_sol_v(self, v_matrix):
        """Update directly v variable from new adjacency matrix."""
        n = len(self.dict_id_c)
        for i, j in product(range(n), range(n)):
            key = (self.id_list[j], self.id_list[i])
            self.instance_model.sol_v[key] = int(v_matrix[i][j])

    def update_w(self, w):
        """Update directly the w param in instance from new w matrix."""
        for i, j in product(range(len(w)), range(len(w[0]))):
            self.instance_model.w[(self.id_list[j], self.id_list[i])] = float(w[i][j])

    def update_obj_place(self, dv):
        """Update the objective for placement with new dv matrix."""
        self.update_dv(dv)
        self.instance_model.obj = sum(
            self.instance_model.sol_u[(i, j)] * self.instance_model.v[(i, j)]
            for i, j in product(self.instance_model.C, self.instance_model.C) if i < j
        ) + sum(
            (1 - self.instance_model.sol_u[(i, j)]
             ) * self.instance_model.v[(i, j)] * self.instance_model.dv[(i, j)]
            for i, j in product(self.instance_model.C, self.instance_model.C) if i < j
        )

    def update_dv(self, dv):
        """Update directly the dv param in instance from new dv matrix."""
        for i, j in product(range(len(dv)), range(len(dv[0]))):
            self.instance_model.dv[(self.id_list[j], self.id_list[i])] = float(dv[i][j])

    def update_size_model(self, new_df_indiv=None, verbose=False):
        """
        Rebuild the abstract & concrete models when the set of containers
        changes (e.g. new/deleted containers).
        """
        if new_df_indiv is not None:
            self.df_indiv = new_df_indiv
        self.mdl = pe.AbstractModel()
        self._build_model()
        self.create_data()
        if verbose:
            self.write_infile(fname=f'./debug_pb{self.pb_number}.lp')
        self.instance_model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    def fill_dual_values(self) -> Dict[Any, float]:
        """
        Extract duals for the 'must_link' constraints after solve().
        Returns a mapping from index (container tuple) to its dual value.
        """
        dual = self.instance_model.dual
        con = getattr(
            self.instance_model, 'must_link_c' if self.pb_number == 1 else 'must_link_n', None
        )
        if con is None:
            self.last_duals = {}
        self.last_duals = {idx: dual[con[idx]] for idx in con}


# ----- rules -----
def _clust_assign_rule(mdl, c):
    return sum(mdl.assign[c, k] for k in mdl.K) == 1


def _open_cluster_rule(mdl, c, k):
    return mdl.assign[c, k] <= mdl.yc[k]


def _max_clusters_rule(mdl):
    return sum(mdl.yc[k] for k in mdl.K) <= mdl.k


def _link_u1_rule(mdl, i, j, k):
    return mdl.u[i, j] >= mdl.assign[i, k] + mdl.assign[j, k] - 1


def _link_u2_rule(mdl, i, j):
    return mdl.u[i, j] <= sum(mdl.assign[i, k] for k in mdl.K)


def _link_u3_rule(mdl, i, j):
    return mdl.u[i, j] <= sum(mdl.assign[j, k] for k in mdl.K)


def _capacity_rule(mdl, node, t):
    return mdl.node_load[node, t] <= mdl.cap[node]


def _open_node(mdl, c, n):
    return mdl.place[c, n] <= mdl.a[n]


def _assignment(mdl, c):
    return sum(mdl.place[c, n] for n in mdl.N) == 1


def _link_v1_rule(mdl, i, j, n):
    return mdl.v[i, j] >= mdl.place[i, n] + mdl.place[j, n] - 1


def _link_v2_rule(mdl, i, j):
    return mdl.v[i, j] <= sum(mdl.place[i, n] for n in mdl.N)


def _link_v3_rule(mdl, i, j):
    return mdl.v[i, j] <= sum(mdl.place[j, n] for n in mdl.N)


def _must_link_c(mdl, i, j):
    uu = mdl.sol_u[(i, j)].value
    if uu == 1:
        return mdl.u[(i, j)] == 1
    else:
        return pe.Constraint.Skip


def _must_link_n(mdl, i, j):
    vv = mdl.sol_v[(i, j)].value
    if vv == 1:
        return mdl.v[(i, j)] == 1
    else:
        return pe.Constraint.Skip


def _min_dissim(mdl):
    """Express the within clusters dissimilarities."""
    return sum(mdl.u[(i, j)] * mdl.w[(i, j)] for i, j in product(mdl.C, mdl.C) if i < j)


def _min_coloc_cluster(mdl):
    """Express the placement minimization objective from clustering."""
    return sum(mdl.sol_u[(i, j)] * mdl.v[(i, j)] for i, j in product(mdl.C, mdl.C) if i < j) + \
        sum((1 - mdl.sol_u[(i, j)]) * mdl.v[(i, j)] * mdl.dv[(i, j)]
            for i, j in product(mdl.C, mdl.C) if i < j)
