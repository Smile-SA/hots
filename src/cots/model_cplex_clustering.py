"""
=========
cots model_cplex
=========

Define the optimization model we have, with its objective, constraints,
variables, and build it from the ``Instance``. Provide all optim model
related methods.
The optimization model description is based on docplex, the Python API of
CPLEX, own by IBM.
"""

import math
import re
import time
from itertools import combinations, product
from typing import Dict, List, Tuple

# from docplex.mp.conflict_refiner import ConflictRefiner
from docplex.mp.linear import LinearExpr
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution

import networkx as nx

import numpy as np

import pandas as pd

from . import init as it
from .clustering import get_far_container


class CPXInstance:
    """
    Class describing the optimization model given to CPLEX.

    Attributes :
    - mdl : DOCPLEX Model instance
    - relax_mdl : linear relaxation of mdl
    - nb_nodes :
    - nb_containers :
    - time_window :
    - nodes_names :
    - containers_names :
    - nodes_data :
    - containers_data :
    - current_sol :
    - max_open_nodes :
    """

    # functions #

    # TODO parallelize all building
    # TODO nice writing in log file
    # TODO optim : adding constraints by batch
    def __init__(self, df_indiv: pd.DataFrame,
                 df_host_meta: pd.DataFrame, nb_clusters: int,
                 dict_id_c: Dict, dict_id_n: Dict,
                 w: np.array = None, u: np.array = None,
                 v: np.array = None, dv: np.array = None,
                 nb_nodes: int = None, pb_number: int = None, verbose: bool = False):
        """Initialize CPXInstance with data in Instance."""
        model_time = time.time()
        self.nb_nodes = df_host_meta[it.host_field].nunique()
        self.nb_containers = df_indiv[it.indiv_field].nunique()
        self.nb_clusters = nb_clusters
        self.time_window = df_indiv[it.tick_field].nunique()

        # Fix max number of nodes (TODO function to evaluate it)
        self.max_open_nodes = nb_nodes or self.nb_nodes

        # Which problem we build :
        # - 0 = placement + clustering
        # - 1 = only placement
        # - 2 = only clustering
        # - 3 = placement from clustering
        self.pb_number = pb_number or 0

        # self.mdl = Model(name=self.get_pb_name(self.pb_number), checker='off')
        self.relax_mdl = Model(name='lp_' + self.get_pb_name(self.pb_number), checker='off')
        # self.mdl = Model(name=self.get_pb_name(self.pb_number), cts_by_name=True)
        # self.relax_mdl = Model(name='lp_' + self.get_pb_name(self.pb_number), cts_by_name=True)
        if verbose:
            it.optim_file.write('Building names ...\n')
        self.build_names()
        if verbose:
            it.optim_file.write('Building variables ...\n')
        self.build_variables()
        if verbose:
            it.optim_file.write('Building data ...\n')
        self.build_data(df_indiv, df_host_meta, dict_id_c, dict_id_n)
        if verbose:
            it.optim_file.write('Building constraints ...\n')
        self.build_constraints(w, u, v)
        if verbose:
            it.optim_file.write('Building objective ...\n')
        self.build_objective(w, u, dv)

        if verbose:
            it.optim_file.write('Building relaxed model ...\n')

        if verbose:
            self.mdl.print_information()
            it.optim_file.write('Building model time : %f s\n\n' % (time.time() - model_time))
        self.write_infile()

    def get_pb_name(self, pb_number: int) -> str:
        """Get the problem name depending on his number."""
        pb_names = ['place-&-clust',
                    'placement',
                    'clustering',
                    'place-f-clust',
                    'clustering_bis']
        return pb_names[pb_number]

    def build_names(self):
        """Build only names variables from nodes and containers."""
        self.nodes_names = np.arange(self.nb_nodes)
        self.containers_names = np.arange(self.nb_containers)

        # Add clusters stuff
        self.clusters_names = np.arange(self.nb_clusters)

    def build_variables(self):
        """Build all model variables."""
        if self.pb_number == 0:
            self.build_clustering_variables()
            self.build_placement_variables()
        elif self.pb_number == 1:
            self.build_placement_variables()
        elif self.pb_number == 2:
            self.build_clustering_variables()
        elif self.pb_number == 3:
            self.build_placement_variables()
        elif self.pb_number == 4:
            self.build_clustering_variables()

    def build_placement_variables(self):
        """Build placement related variables."""
        idx = [(c, n) for c in self.containers_names for n in self.nodes_names]
        # placement variables : if container c is on node n
        self.relax_mdl.x = self.relax_mdl.continuous_var_dict(
            idx, name=lambda k: 'x_%d,%d' % (k[0], k[1]),
            lb=0, ub=1)

        ida = list(self.nodes_names)
        # opened variables : if node n is used or not
        self.relax_mdl.a = self.relax_mdl.continuous_var_dict(
            ida, name=lambda k: 'a_%d' % k, lb=0, ub=1
        )

        # variables for max diff consumption (global delta)
        self.relax_mdl.delta = self.relax_mdl.continuous_var(name='delta')

        idv = [(c1, c2) for (c1, c2) in combinations(self.containers_names, 2)]
        # coloc variables : if c1 and c2 are in the same node
        self.relax_mdl.v = self.relax_mdl.continuous_var_dict(
            idv, name=lambda k: 'v_%d,%d' % (k[0], k[1]),
            lb=0, ub=1
        )

        # specific node coloc variables : if c1 and c2 are in node n
        idxv = [(c1, c2, n) for (c1, c2) in idv for n in self.nodes_names]
        self.relax_mdl.xv = self.relax_mdl.continuous_var_dict(
            idxv, name=lambda k: 'xv_%d,%d,%d' % (k[0], k[1], k[2]),
            lb=0, ub=1
        )

    def build_clustering_variables(self):
        """Build clustering related variables."""
        idy = [
            (c, k) for c in self.containers_names for k in self.clusters_names]
        # assignment variables : if container c is on cluster k
        self.relax_mdl.y = self.relax_mdl.continuous_var_dict(
            idy, name=lambda k: 'y_%d,%d' % (k[0], k[1]))

        idk = list(self.clusters_names)
        # used cluster variables
        self.relax_mdl.b = self.relax_mdl.continuous_var_dict(
            idk, name=lambda k: 'b_%d' % k
        )

        idu = [(c1, c2) for (c1, c2) in combinations(self.containers_names, 2)]
        # coloc variables : if containers c1 and c2 are in same cluster
        self.relax_mdl.u = self.relax_mdl.continuous_var_dict(
            idu, name=lambda k: 'u_%d,%d' % (k[0], k[1])
        )

        # specific cluster coloc variables : if c1 and c2 are in cluster k
        idyu = [(c1, c2, k) for (c1, c2) in idu for k in self.clusters_names]
        self.relax_mdl.yu = self.relax_mdl.continuous_var_dict(
            idyu, name=lambda k: 'yu_%d,%d,%d' % (k[0], k[1], k[2])
        )

    def build_data(self, df_indiv: pd.DataFrame,
                   df_host_meta: pd.DataFrame,
                   dict_id_c: Dict, dict_id_n: Dict):
        """Build all model data from instance."""
        self.containers_data = {}
        self.nodes_data = {}
        for c in self.containers_names:
            self.containers_data[c] = df_indiv.loc[
                df_indiv[it.indiv_field] == dict_id_c[c],
                [it.tick_field, it.metrics[0]]].to_numpy()
        for n in self.nodes_names:
            self.nodes_data[n] = df_host_meta.loc[
                df_host_meta[it.host_field] == dict_id_n[n],
                [it.metrics[0]]].to_numpy()[0]

    def update_data(self, df_indiv, dict_id_c):
        """Update the data used by the model."""
        for c in self.containers_names:
            self.containers_data[c] = df_indiv.loc[
                df_indiv[it.indiv_field] == dict_id_c[c],
                [it.tick_field, it.metrics[0]]].to_numpy()

    def build_constraints(self, w, u, v):
        """Build all model constraints."""
        if self.pb_number == 0:
            self.placement_constraints()
            self.clustering_constraints(self.nb_clusters, w)
        elif self.pb_number == 1:
            self.placement_constraints()
        elif self.pb_number == 2:
            self.clustering_constraints(self.nb_clusters, w)
            self.add_adjacency_clust_constraints(u)
        elif self.pb_number == 3:
            self.placement_constraints()
            self.add_adjacency_place_constraints(v)
        elif self.pb_number == 4:
            self.clustering_constraints(self.nb_clusters, w)
            self.add_cannotlink_constraints(u)

    def clustering_constraints(self, nb_clusters, w):
        """Build the clustering related constraints."""
        # Cluster assignment constraint
        self.relax_mdl.add_constraints(
            (self.relax_mdl.scal_prod(
                [self.relax_mdl.y[c, k] for k in self.clusters_names], 1
            ) == 1 for c in self.containers_names),
            ('cluster_assign_%d' % c for c in self.containers_names))

        # Open cluster
        self.relax_mdl.add_constraints(
            ((self.relax_mdl.y[c, k] - self.relax_mdl.b[k]
              ) <= 0 for (c, k) in product(self.containers_names, self.clusters_names)),
            ('open_cluster_%d_%d' % (c, k) for (c, k) in product(
                self.containers_names, self.clusters_names))
        )

        # Number of clusters
        self.relax_mdl.add_constraint(
            self.relax_mdl.sum(
                self.relax_mdl.b[k] for k in self.clusters_names) <= self.nb_clusters,
            'nb_clusters'
        )

    def add_adjacency_clust_constraints(self, u):
        """Add constraints fixing u variables from adjacency matrice."""
        self.adj_constr_relax = []
        self.adj_constr_relax.extend(self.relax_mdl.add_constraints(
            ((self.relax_mdl.u[i, j] == 1) for (i, j) in combinations(
                self.containers_names, 2) if u[i, j]),
            ('mustLinkC_%d_%d' % (i, j) for (i, j) in combinations(
                self.containers_names, 2) if u[i, j])
        ))

    def update_adjacency_clust_constraints(self, u):
        """Update constraints fixing u variables from new adjacency matrix."""
        self.relax_mdl.remove_constraints(self.adj_constr_relax)
        self.add_adjacency_clust_constraints(u)

    def placement_constraints(self):
        """Build the placement problem related constraints."""
        # Capacity constraints
        # TODO if several metrics ?
        self.relax_mdl.add_constraints(
            (self.relax_mdl.sum(
                self.relax_mdl.x[c, n] * self.containers_data[c][t][1]
                for c in self.containers_names
            ) <= self.nodes_data[n][0] for (n, t) in product(
                self.nodes_names, range(self.time_window))),
            ('capacity_%d_%d' % (n, t) for (n, t) in product(
                self.nodes_names, range(self.time_window)))
        )

        # Open node
        self.relax_mdl.add_constraints(
            ((self.relax_mdl.x[c, n] - self.relax_mdl.a[n]
              ) <= 0 for (c, n) in product(self.containers_names, self.nodes_names)),
            ('open_node_%d_%d' % (c, n) for (c, n) in product(
                self.containers_names, self.nodes_names))
        )

        # Container assignment constraint (1 and only 1 x[c,_] = 1 for all c)
        self.relax_mdl.add_constraints(
            (self.relax_mdl.scal_prod(
                [self.relax_mdl.x[c, n] for n in self.nodes_names], 1
            ) == 1 for c in self.containers_names),
            ('assign_%d' % c for c in self.containers_names))

    def add_adjacency_place_constraints(self, v):
        """Add constraints fixing v variables from adjacency matrice."""
        # Constraints fixing xv(i, j, n) and mustlinkA constraints
        # TODO replace because too many variables
        self.adj_constr_relax = []
        self.adj_constr_relax.extend(self.relax_mdl.add_constraints(
            ((self.relax_mdl.v[i, j] == 1) for (i, j) in combinations(
                self.containers_names, 2) if v[i, j]),
            ('mustLinkA_%d_%d' % (i, j) for (i, j) in combinations(
                self.containers_names, 2) if v[i, j])
        ))

    def update_place_constraints(self, v):
        """Update placement constraints with new data."""
        for (i, j) in combinations(self.containers_names, 2):
            ct = self.mdl.get_constraint_by_name(
                'mustLinkA_' + str(j) + '_' + str(i)
            )
            if ct is None and v[i, j]:
                self.mdl.add_constraint(
                    self.mdl.v[i, j] == 1,
                    'mustLinkA_' + str(j) + '_' + str(i)
                )
                self.relax_mdl.add_constraint(
                    self.relax_mdl.v[i, j] == 1,
                    'mustLinkA_' + str(j) + '_' + str(i)
                )
            elif ct is not None and not v[i, j]:
                self.mdl.remove_constraint(
                    'mustLinkA_' + str(j) + '_' + str(i)
                )
                self.relax_mdl.remove_constraint(
                    'mustLinkA_' + str(j) + '_' + str(i)
                )

    def update_adjacency_place_constraints(self, v):
        """Update constraints fixing u variables from new adjacency matrix."""
        self.relax_mdl.remove_constraints(self.adj_constr_relax)
        self.add_adjacency_place_constraints(v)

    def add_cannotlink_constraints(self, u):
        """Add constraints fixing u variables from adjacency matrice."""
        # TODO replace because too many variables
        for (i, j) in combinations(self.containers_names, 2):
            if not u[i, j]:
                for k in self.clusters_names:
                    self.mdl.add_constraint(
                        self.mdl.yu[i, j, k] <= self.mdl.y[i, k],
                        'linear1_yu' + str(i) + '_' + str(j) + '_' + str(k)
                    )
                    self.mdl.add_constraint(
                        self.mdl.yu[i, j, k] <= self.mdl.y[j, k],
                        'linear2_yu' + str(i) + '_' + str(j) + '_' + str(k)
                    )
                    self.mdl.add_constraint(
                        self.mdl.y[i, k] + self.mdl.y[j, k] - self.mdl.yu[i, j, k] <= 1,
                        'linear3_yu' + str(i) + '_' + str(j) + '_' + str(k)
                    )

                # Constraints fixing u
                self.mdl.add_constraint(
                    self.mdl.u[i, j] == self.mdl.sum(
                        self.mdl.yu[i, j, k] for k in self.clusters_names),
                    'fix_u' + str(i) + '_' + str(j)
                )
                self.mdl.add_constraint(
                    self.mdl.u[i, j] == 0,
                    'cannotLinkC_' + str(j) + '_' + str(i)
                )

    def build_objective(self, w, u, dv):
        """Build objective."""
        if self.pb_number == 0:
            # Minimize delta + sum(w*y)
            self.mdl.minimize(
                self.mdl.delta
                + self.mdl.sum(
                    self.mdl.z[i, j] for (i, j) in combinations(
                        self.containers_names, 2
                    )
                )
            )
        elif self.pb_number == 1:
            # Minimize delta (max(conso_n_t - mean_n))
            self.mdl.minimize(
                self.mdl.sum(
                    self.mdl.z[i, j] for (i, j) in combinations(
                        self.containers_names, 2
                    ))
            )
        elif self.pb_number == 2:
            # Only clustering
            self.relax_mdl.minimize(self.relax_mdl.sum(
                w[i, j] * self.relax_mdl.u[i, j] for (i, j) in combinations(
                    self.containers_names, 2
                )
            ))
        elif self.pb_number == 3:
            # Placement via clusters profiles
            self.relax_mdl.minimize(
                self.relax_mdl.sum(
                    u[i, j] * self.relax_mdl.v[i, j] for (i, j) in combinations(
                        self.containers_names, 2
                    )
                ) + self.relax_mdl.sum(
                    (1 - u[i, j]) * self.relax_mdl.v[i, j] * dv[i, j]
                    for (i, j) in combinations(
                        self.containers_names, 2
                    )
                )
            )
        elif self.pb_number == 4:
            # Only clustering
            self.mdl.maximize(
                self.mdl.sum(
                    (1 - self.mdl.u[i, j]) * w[i, j] for (i, j) in combinations(
                        self.containers_names, 2
                    )
                )
            )

    def update_obj_clustering(self, w):
        """Remove and re-create objective for clustering model."""
        self.relax_mdl.remove_objective()
        self.relax_mdl.minimize(self.relax_mdl.sum(
            w[i, j] * self.relax_mdl.u[i, j] for (i, j) in combinations(
                self.containers_names, 2
            )
        ))

    def update_obj_placement(self, u, v, dv):
        """Remove and re-create objective for placement model."""
        self.relax_mdl.remove_objective()
        self.relax_mdl.minimize(
            self.relax_mdl.sum(
                u[i, j] * self.relax_mdl.v[i, j] for (i, j) in combinations(
                    self.containers_names, 2
                )
            ) + self.relax_mdl.sum(
                (1 - u[i, j]) * self.relax_mdl.v[i, j] * dv[i, j]
                for (i, j) in combinations(
                    self.containers_names, 2
                )
            )
        )

    def set_x_from_df(self, df_indiv: pd.DataFrame,
                      dict_id_c: Dict, dict_id_n: Dict):
        """Add feasible solution from allocation in instance."""
        start_sol = {}
        for c in self.containers_names:
            node_c_id = df_indiv.loc[
                df_indiv[it.indiv_field] == dict_id_c[c], it.host_field
            ].to_numpy()[0]
            node_c = list(dict_id_n.keys())[list(
                dict_id_n.values()).index(node_c_id)]
            for n in self.nodes_names:
                start_sol[self.mdl.x[c, n]] = 0
            start_sol[self.mdl.x[c, node_c]] = 1
            if not self.mdl.a[node_c] in start_sol:
                start_sol[self.mdl.a[node_c]] = 1
        self.current_sol = SolveSolution(self.mdl, start_sol)
        self.mdl.add_mip_start(self.current_sol)

    def solve(self, my_mdl: Model, verbose: bool = False):
        """Solve the given problem."""
        if not my_mdl.solve(
            clean_before_solve=True, log_output=verbose
        ):
            it.optim_file.write('*** Problem has no solution ***')
            it.optim_file.write(my_mdl.solve_details.status)
        else:
            it.optim_file.write('*** Model %s solved as function:' % self.pb_number)
            if verbose:
                my_mdl.print_solution()
                my_mdl.report()
            it.optim_file.write('%f\n' % my_mdl.objective_value)

    def print_heuristic_solution_a(self):
        """Print the heuristic solution (variables a)."""
        it.optim_file.write('Values of variables a :')
        for n in self.nodes_names:
            it.optim_file.write(self.mdl.a[n], self.current_sol.get_value(
                self.mdl.a[n]))

    def print_sol_infos_heur(self, total_time: int = 6):
        """Print information about heuristic solution."""
        it.optim_file.write('Infos about heuristics solution ...')

        means_ = np.zeros(len(self.nodes_names))
        vars_ = np.zeros(len(self.nodes_names))
        for n in self.nodes_names:

            # Compute mean in each node
            for c in self.containers_names:
                if self.current_sol.get_value(self.mdl.x[c, n]):
                    for t in range(total_time):
                        means_[n] += self.containers_data[c][t][1]
            means_[n] = means_[n] / total_time
            it.optim_file.write('Mean of node %d : %f' % (n, means_[n]))

            # Compute variance in each node
            for t in range(total_time):
                total_t = 0.0
                for c in self.containers_names:
                    if self.current_sol.get_value(self.mdl.x[c, n]):
                        total_t += self.containers_data[c][t][1]
                vars_[n] += math.pow(total_t - means_[n], 2)
            vars_[n] = vars_[n] / total_time
            it.optim_file.write('Variance of node %d : %f' % (n, vars_[n]))

    # We can get dual values only for LP
    def get_max_dual(self):
        """Get the constraint with the highest dual variable value."""
        if self.relax_mdl is None:
            it.optim_file.write("*** Linear Relaxation does not exist : we can't get\
                dual values ***")
        else:
            ct_max = None
            dual_max = 0.0
            for ct in self.relax_mdl.iter_linear_constraints():
                if ct.dual_value > dual_max:
                    ct_max = ct
                    dual_max = ct.dual_value
            it.optim_file.write(ct_max, dual_max)

    # Expr total conso CPU in node at t
    def conso_n_t(self, node, t: int) -> LinearExpr:
        """Express the total consumption of node at time t."""
        return self.mdl.sum(
            (self.mdl.x[c, node] * self.containers_data[c][t][1])
            for c in self.containers_names)

    # Expr mean CPU in node
    def mean(self, node: int) -> LinearExpr:
        """Express the mean consumption of node."""
        return (self.mdl.sum(
            self.conso_n_t(node, t) for t in range(self.time_window)
        ) / self.time_window)

    def mean_all_nodes(self, total_time: int) -> LinearExpr:
        """Express the mean consumption of all nodes."""
        return (self.mdl.sum(
            self.mean(node, total_time)
            for node in self.nodes_names) / self.nb_nodes)

    # Expr variance CPU in node
    def var(self, node: int, total_time: int) -> LinearExpr:
        """Express the variance of consumption of node."""
        return (self.mdl.sum(
            (self.conso_n_t(node, t) - self.mean(node, total_time))**2
            for t in range(total_time)) / total_time)

    # Expr max CPU in node
    def max_n(self, node: int, total_time: int) -> LinearExpr:
        """Express the maximum consumption of node."""
        return (self.mdl.max(
            [self.conso_n_t(node, t) for t in range(total_time)]))

    def write_infile(self):
        """Write the problem in LP file."""
        # TODO write these files in data folder
        if self.pb_number == 0:
            self.relax_mdl.export_as_lp(path='./lp_place_w_clust.lp')
        elif self.pb_number == 1:
            self.relax_mdl.export_as_lp(path='./lp_placement.lp')
        elif self.pb_number == 2:
            self.relax_mdl.export_as_lp(path='./lp_clustering.lp')
        elif self.pb_number == 3:
            self.relax_mdl.export_as_lp(path='./lp_place_f_clust.lp')
        elif self.pb_number == 4:
            self.relax_mdl.export_as_lp(path='./clustering_bis.lp')


# Functions related to CPLEX #

def get_obj_value_heuristic(df_indiv: pd.DataFrame,
                            t_min: int = None,
                            t_max: int = None) -> Tuple[int, float]:
    """Get objective value of current solution (max delta)."""
    t_min = t_min or df_indiv[it.tick_field].min()
    t_max = t_max or df_indiv[it.tick_field].max()
    df_indiv = df_indiv[
        (df_indiv[it.tick_field] >= t_min)
        & (df_indiv[it.tick_field] <= t_max)]
    df_indiv.reset_index(drop=True, inplace=True)
    obj_val = 0.0
    for n, n_data in df_indiv.groupby(
            [it.host_field]):
        max_n = 0.0
        min_n = -1.0
        # mean_n = 0.0
        # nb_t = 0
        for t, nt_data in n_data.groupby(it.tick_field):
            if nt_data[it.metrics[0]].sum() > max_n:
                max_n = nt_data[it.metrics[0]].sum()
            if (nt_data[it.metrics[0]].sum() < min_n) or (
                    min_n < 0.0):
                min_n = nt_data[it.metrics[0]].sum()
        #     mean_n += nt_data[it.metrics[0]].sum()
        #     nb_t += 1
        # delta_n = max_n - (mean_n / nb_t)
        delta_n = max_n - min_n
        if obj_val < delta_n:
            obj_val = delta_n
    return (df_indiv[it.host_field].nunique(), obj_val)


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


def print_constraints(mdl: Model):
    """Print all the constraints."""
    for ct in mdl.iter_constraints():
        it.optim_file.write(ct)


def print_all_dual(mdl: Model, nn_only: bool = True,
                   names: List = None):
    """Print dual values associated to constraints."""
    if nn_only:
        it.optim_file.write('Display non-zero dual values')
    else:
        it.optim_file.write('Display all dual values')
    for ct in mdl.iter_linear_constraints():
        if not nn_only:
            it.optim_file.write(ct, ct.dual_value)
        elif ct.dual_value > 0 and nn_only and names is None:
            it.optim_file.write(ct, ct.dual_value)
        elif ct.dual_value > 0 and True in (name in ct.name for name in names):
            it.optim_file.write(ct, ct.dual_value)


def fill_constraints_dual_values(mdl: Model, names: List) -> Dict:
    """Fill specific constraints with their dual values."""
    constraints_dual_values = {}
    for ct in mdl.iter_linear_constraints():
        if True in (name in ct.name for name in names):
            constraints_dual_values[ct.name] = ct.dual_value
    return constraints_dual_values


def get_max_dual(mdl: Model, names: List = None):
    """Get the max dual value associated to some constraints."""
    max_dual = 0.0
    for ct in mdl.iter_linear_constraints():
        if (ct.dual_value > max_dual) and (
            (True in (name in ct.name for name in names)) or (
                names is None)):
            max_dual = ct.dual_value
    return max_dual


def dual_changed(mdl: Model, names: List,
                 prev_dual: float, tol: float) -> bool:
    """Check if dual values increase since last eval."""
    for ct in mdl.iter_linear_constraints():
        if (ct.dual_value > (prev_dual + tol * prev_dual)) and True in (
                name in ct.name for name in names):
            return True
    return False


def get_conflict_graph(mdl: Model, constraints_dual_values: Dict, tol: float):
    """Build conflict graph from comapring dual variables."""
    conflict_graph = nx.Graph()
    for ct in mdl.iter_linear_constraints():
        if (ct.name in constraints_dual_values) and (
                constraints_dual_values[ct.name] > 0.0):
            if (ct.dual_value > (
                    constraints_dual_values[ct.name]
                    + tol * constraints_dual_values[ct.name])) or (
                        ct.dual_value > tol * mdl.objective_value
            ):
                indivs = re.findall(r'\d+\.*', ct.name)
                conflict_graph.add_edge(indivs[0], indivs[1], weight=ct.dual_value)
    return conflict_graph


def get_moving_containers_clust(mdl: Model, constraints_dual_values: Dict,
                                tol: float, tol_move: float, nb_containers: int,
                                dict_id_c: Dict, df_clust: pd.DataFrame, profiles: np.array
                                ) -> Tuple[List, int, int, int, int]:
    """Get the list of moving containers from constraints dual values."""
    mvg_containers = []
    conflict_graph = get_conflict_graph(mdl, constraints_dual_values, tol)

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
def get_moving_containers_place(mdl: Model, constraints_dual_values: Dict,
                                tol: float, tol_move: float, nb_containers: int,
                                working_df: pd.DataFrame, dict_id_c: Dict
                                ) -> Tuple[List, int, int, int, int]:
    """Get the list of moving containers from constraints dual values."""
    mvg_containers = []
    conflict_graph = get_conflict_graph(mdl, constraints_dual_values, tol)

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


def perform_k_max_cut(graph: nx.Graph(), k: int) -> List:
    """Perform greedy k-max-cut algo on conflict graph."""
    cut = []
    cut_val = 0.0
    sets = [[] for i in range(k)]
    for indiv, data in graph.nodes(data=True):
        best_set = 0
        best_cut_val = 0.0
        best_cut = []
        for k_i in range(k):
            k_cut_val = 0
            temp_cut = []
            for k_j in range(k):
                if k_j == k_i:
                    continue
                for m_k in sets[k_j]:
                    if m_k in graph.adj[indiv]:
                        k_cut_val += graph.get_edge_data(indiv, m_k)['weight']
                        temp_cut.append(graph.get_edge_data(indiv, m_k)['ct'])
            if k_cut_val > best_cut_val:
                best_cut_val = k_cut_val
                best_set = k_i
                best_cut = temp_cut
        sets[best_set].append(indiv)
        cut.extend(best_cut)
        cut_val += best_cut_val
    return cut


def set_u_constraints_rm(constraints_rm: List, u: np.array):
    """Modify u matrix from constraints removed in model."""
    for ct in constraints_rm:
        indivs = re.findall(r'\d+\.*', ct.name)
        u[int(indivs[0])][int(indivs[1])] = 0
        u[int(indivs[1])][int(indivs[0])] = 0
    return u
