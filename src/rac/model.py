"""
=========
rac model
=========

Define the optimization model we have, with its objective, constraints,
variables, and build it from the ``Instance``. Provide all optim model
related methods.
The optimization model description is based on docplex, the Python API of
CPLEX, own by IBM.
"""

import math
import time
from itertools import combinations
from typing import Dict, List

from docplex.mp.linear import LinearExpr
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution

import numpy as np

from .init import metrics
from .instance import Instance
from .node import get_mean_consumption_node


# TODO add KPIs ?


class CPXInstance:
    """
    Class describing the optimization model given to CPLEX.

    Attributes :
    - mdl : DOCPLEX Model instance
    - relax_mdl : linear relaxation of mdl
    - nb_nodes :
    - nb_containers :
    - nodes_names :
    - containers_names :
    - nodes_data :
    - containers_data :
    - current_sol :
    - max_open_nodes :
    """

    # functions #

    # TODO parallelize all building
    def __init__(self, my_instance: Instance):
        """Initialize CPXInstance with data in Instance."""
        model_time = time.time()
        print('Building of cplex model ...')
        self.nb_nodes = my_instance.df_nodes['machine_id'].nunique()
        self.nb_containers = my_instance.df_containers['container_id'].nunique(
        )

        # Fix max number of nodes (TODO function to evaluate it)
        self.max_open_nodes = my_instance.nb_nodes

        self.mdl = Model(name='allocation')
        self.build_names()
        self.build_variables()
        self.build_data(my_instance)
        self.build_constraints(my_instance.sep_time)
        self.build_objective(my_instance)

        self.relax_mdl = make_relaxed_model(self.mdl)

        # Init solution for mdl with initial placement
        self.set_x_from_df(my_instance)

        self.mdl.print_information()

        print('Building model time : ', time.time() - model_time)

        self.mdl.export_as_lp(path='./allocation.lp')

    def build_names(self):
        """Build only names variables from nodes and containers."""
        self.nodes_names = np.arange(self.nb_nodes)
        self.containers_names = np.arange(self.nb_containers)

    def build_variables(self):
        """Build all model variables."""
        idx = [(c, n) for c in self.containers_names for n in self.nodes_names]
        self.mdl.x = self.mdl.binary_var_dict(
            idx, name=lambda k: 'x_%d,%d' % (k[0], k[1]))

        ida = list(self.nodes_names)
        self.mdl.a = self.mdl.binary_var_dict(ida, name=lambda k: 'a_%d' % k)

        # variables for max diff consumption (global delta)
        # self.mdl.delta = self.mdl.continuous_var(name='delta')

        # variables for max diff consumption (n delta)
        self.mdl.delta = self.mdl.continuous_var_dict(
            ida, name=lambda k: 'delta_%d' % k)

    def build_data(self, my_instance: Instance):
        """Build all model data from instance."""
        self.containers_data = {}
        self.nodes_data = {}
        for c in self.containers_names:
            self.containers_data[c] = my_instance.df_containers.loc[
                my_instance.
                df_containers['container_id'] == my_instance.dict_id_c[c],
                ['timestamp', 'cpu', 'mem']].to_numpy()
        for n in self.nodes_names:
            self.nodes_data[n] = my_instance.df_nodes_meta.loc[
                my_instance.
                df_nodes_meta['machine_id'] == my_instance.dict_id_n[n],
                ['cpu', 'mem']].to_numpy()[0]

    def build_constraints(self, total_time: int):
        """Build all model constraints."""
        for node in self.nodes_names:
            for t in range(total_time):
                i = 0
                for i in range(1, len(metrics) + 1):
                    # Capacity constraint
                    self.mdl.add_constraint(self.mdl.sum(
                        self.mdl.x[c, node] * self.containers_data[c][t][i]
                        for c in self.
                        containers_names) <= self.nodes_data[node][i - 1],
                        metrics[i - 1] + 'capacity_' + str(node) + '_' + str(t))

            # Assign constraint (x[c,n] = 1 => a[n] = 1)
            # self.mdl.add_constraint(self.mdl.a[node] >= (self.mdl.sum(
            #     self.mdl.x[c, node]
            # for c in self.containers_names) / len(self.containers_names)
            # ), 'open_a_'+str(node))

            # more constraints, but easier
            for c in self.containers_names:
                self.mdl.add_constraint(
                    self.mdl.x[c, node] <= self.mdl.a[node],
                    'open_a_' + str(node))

        # Container assignment constraint (1 and only 1 x[c,_] = 1 for all c)
        for container in self.containers_names:
            self.mdl.add_constraint(self.mdl.sum(
                self.mdl.x[container, node] for node in self.nodes_names) == 1,
                'assign_' + str(container))

        # Assign delta to diff cons - mean_cons
        for node in self.nodes_names:
            expr_n = self.mean(node, total_time)
            for t in range(total_time):
                self.mdl.add_constraint(self.conso_n_t(
                    node, t) - expr_n <= self.mdl.delta[node],
                    'delta_' + str(node) + '_' + str(t))
                self.mdl.add_constraint(
                    expr_n - self.
                    conso_n_t(node, t) <= self.mdl.delta[node],
                    'inv-delta_' + str(node) + '_' + str(t))

        # Constraint the number of open servers
        self.mdl.add_constraint(self.mdl.sum(
            self.mdl.a[n] for n in self.
            nodes_names) == self.max_open_nodes, 'max_nodes')

    def build_objective(self, my_instance: Instance):
        """Build objective."""
        # Minimize sum of a[n] (number of open nodes)
        # self.mdl.minimize(self.mdl.sum(
        #     self.mdl.a[n] for n in self.nodes_names))

        # Minimize sum of v[n] (variance)
        # self.mdl.minimize(self.mdl.sum(
        #     self.var(n, total_time) for n in self.nodes_names))

        # Minimize delta (max(conso_n_t - mean_n))
        # self.mdl.minimize(self.mdl.delta)

        # Minimize sum(delta_n*a_n)
        self.mdl.minimize((
            self.mdl.sum(
                self.mdl.delta[n] for n in self.nodes_names
            ) + self.mdl.sum(self.mdl.a[n] for n in self.nodes_names)
        ))

    def set_x_from_df(self, my_instance: Instance):
        """Add feasible solution from allocation in instance."""
        start_sol = {}
        for c in self.containers_names:
            node_c_id = my_instance.df_containers.loc[
                my_instance.
                df_containers['container_id'] == my_instance.
                dict_id_c[c], 'machine_id'].to_numpy()[0]
            node_c = list(my_instance.dict_id_n.keys())[list(
                my_instance.dict_id_n.values()).index(node_c_id)]
            for n in self.nodes_names:
                start_sol[self.mdl.x[c, n]] = 0
            start_sol[self.mdl.x[c, node_c]] = 1
            if not self.mdl.a[node_c] in start_sol:
                start_sol[self.mdl.a[node_c]] = 1
        self.current_sol = SolveSolution(self.mdl, start_sol)
        self.mdl.add_mip_start(self.current_sol)
        # self.current_sol.print_mst()
        # self.add_constraint_heuristic()

    def get_obj_value_heuristic(self, my_instance: Instance):
        """Get objective value of heuristic solution (max delta)."""
        # TODO adaptative ... (retrieve expr from obj ?)
        obj_val = 0.0
        nb_open_nodes = 0

        for n in self.nodes_names:
            if self.current_sol.get_value(self.mdl.a[n]):
                max_n = 0.0
                mean_n = get_mean_consumption_node(
                    my_instance.df_nodes.loc[
                        my_instance.df_nodes['timestamp'] <= my_instance.
                        sep_time
                    ], my_instance.dict_id_n[n]
                )
                for t in range(my_instance.sep_time):
                    total_t = 0.0
                    for c in self.containers_names:
                        if self.current_sol.get_value(self.mdl.x[c, n]):
                            total_t += self.containers_data[c][t][1]
                    if total_t > max_n:
                        max_n = total_t
                obj_val += max_n - mean_n
                nb_open_nodes += 1
        print('Objective value : ', obj_val + nb_open_nodes)

    def solve(self, my_mdl: Model):
        """Solve the problem."""
        if not my_mdl.solve(log_output=True):
            print('*** Problem has no solution ***')
        else:
            print('*** Model solved as function:')
            my_mdl.print_solution()
            my_mdl.report()
            print(my_mdl.objective_value)
            # my_mdl.report_kpis()

    def solve_relax(self):
        """Solve the linear relaxation of the problem."""
        if not self.relax_mdl.solve(log_output=True):
            print('*** Problem has no solution ***')
        else:
            print('*** Model solved as function:')
            self.relax_mdl.print_solution()
            # self.relax_mdl.report_kpis()

    def print_heuristic_solution_a(self):
        """Print the heuristic solution (variables a)."""
        print('Values of variables a :')
        for n in self.nodes_names:
            print(self.mdl.a[n], self.current_sol.get_value(
                self.mdl.a[n]))

    def print_sol_infos_heur(self, total_time: int = 6):
        """Print information about heuristic solution."""
        print('Infos about heuristics solution ...')

        means_ = np.zeros(len(self.nodes_names))
        vars_ = np.zeros(len(self.nodes_names))
        for n in self.nodes_names:

            # Compute mean in each node
            for c in self.containers_names:
                if self.current_sol.get_value(self.mdl.x[c, n]):
                    for t in range(total_time):
                        means_[n] += self.containers_data[c][t][1]
            means_[n] = means_[n] / total_time
            print('Mean of node %d : %f' % (n, means_[n]))

            # Compute variance in each node
            for t in range(total_time):
                total_t = 0.0
                for c in self.containers_names:
                    if self.current_sol.get_value(self.mdl.x[c, n]):
                        total_t += self.containers_data[c][t][1]
                vars_[n] += math.pow(total_t - means_[n], 2)
            vars_[n] = vars_[n] / total_time
            print('Variance of node %d : %f' % (n, vars_[n]))

    # We can get dual values only for LP
    def get_max_dual(self):
        """Get the constraint with the highest dual variable value."""
        if self.relax_mdl is None:
            print("*** Linear Relaxation does not exist : we can't get\
                dual values ***")
        else:
            self.solve(self.relax_mdl)
            ct_max = None
            dual_max = 0.0
            for ct in self.relax_mdl.iter_linear_constraints():
                if ct.dual_value > dual_max:
                    ct_max = ct
                    dual_max = ct.dual_value
            print(ct_max, dual_max)

    def add_constraint_heuristic(self,
                                 container_grouped: List, instance: Instance):
        """Add constraints from heuristic."""
        # for i_c1 in range(len(self.containers_names)):
        #     c1 = self.containers_names[i_c1]
        #     for i_c2 in range(i_c1+1, len(self.containers_names)):
        #         c2 = self.containers_names[i_c2]
        #         for n in self.nodes_names:
        #             if (not (c1 == c2)) and (self.current_sol.get_value(self.
        # mdl.x[c1, n])) and (self.current_sol.get_value(self.mdl.x[c2, n])):

        # must-link container-container
        # for n_i in self.nodes_names:
        #     self.mdl.add_constraint(
        #         self.mdl.x[c1, n_i] - self.mdl.x[c2, n_i] == 0, 'mustLink_'
        # +str(c1)+'_'+str(c2))

        # fix x container-node
        # self.mdl.add_constraint(
        #     self.mdl.x[c1, n] == 1, 'fix_x_'+str(c1)+'_'+str(n))
        # self.mdl.add_constraint(
        #     self.mdl.x[c2, n] == 1, 'fix_x_'+str(c2)+'_'+str(n))

        # must-link container-container only for pairwise clusters
        for list_containers in container_grouped:
            for (c1, c2) in combinations(list_containers, 2):
                # TODO not ideal : change grouped containers saving in
                # heuristic
                if instance.get_node_from_container(
                        c1) == instance.get_node_from_container(c2):
                    c1 = [k for k, v in instance.
                          dict_id_c.items() if v == c1][0]
                    c2 = [k for k, v in instance.
                          dict_id_c.items() if v == c2][0]
                    for n_i in self.nodes_names:
                        self.mdl.add_constraint(
                            self.mdl.x[c1, n_i] - self.mdl.x[c2, n_i] == 0,
                            'mustLink_' + str(c1) + '_' + str(c2))

        # Update the linear relaxation
        self.relax_mdl = make_relaxed_model(self.mdl)

    # TODO not working since problem_dir deleted
    # def export_mdls_lp(self):
    #     self.mdl.export_as_lp(path=problem_dir)
    #     self.relax_mdl.export_as_lp(path=problem_dir)

    # Expr total conso CPU in node at t
    def conso_n_t(self, node, t: int) -> LinearExpr:
        """Express the total consumption of node at time t."""
        return self.mdl.sum(
            (self.mdl.x[c, node] * self.containers_data[c][t][1])
            for c in self.containers_names)

    # Expr mean CPU in node
    def mean(self, node: int, total_time: int) -> LinearExpr:
        """Express the mean consumption of node."""
        return (self.mdl.sum(
            self.conso_n_t(node, t) for t in range(total_time)
        ) / total_time)

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


# Functions related to CPLEX #

def print_constraints(mdl: Model):
    """Print all the constraints."""
    for ct in mdl.iter_constraints():
        print(ct)


def print_all_dual(mdl: Model, nn_only: bool = True):
    """Print dual values associated to constraints."""
    if nn_only:
        print('Display non-zero dual values')
    else:
        print('Display all dual values')
    for ct in mdl.iter_linear_constraints():
        if not nn_only:
            print(ct, ct.dual_value)
        elif ct.dual_value > 0 and nn_only:
            print(ct, ct.dual_value)


def print_non_user_constraint(mdl: Model):
    """Print constraints not added by the user."""
    print('*** Print non user constraints ***', mdl.name)
    for ct in mdl.iter_constraints():
        if not ct.has_user_name():
            print(ct, ct.is_generated())


def transf_vars(mdl: Model, relax_mdl: Model, ctn_map: Dict) -> Dict:
    """Transfer variables from original model to relaxed model."""
    var_mapping = {}
    continuous = relax_mdl.continuous_vartype
    for v in mdl.iter_variables():
        if not v.is_generated():
            # if v has type semixxx, set lB to 0
            cpx_code = v.vartype.get_cplex_typecode()
            if cpx_code in {'N', 'S'}:
                rx_lb = 0
            else:
                rx_lb = v.lb
            copied_var = relax_mdl._var(continuous, rx_lb, v.ub, v.name)
            var_ctn = v._container
            if var_ctn:
                copied_var._container = ctn_map.get(var_ctn)
            var_mapping[v] = copied_var
    return var_mapping


def transf_constraints(mdl: Model, relax_mdl: Model, var_mapping: Dict) -> List:
    """Transfer non-logical constraints from original model to relaxed one."""
    unrelaxables = []
    for ct in mdl.iter_constraints():
        if not ct.is_generated():
            if ct.is_logical():
                unrelaxables.append(ct)
            else:
                copied_ct = ct.copy(relax_mdl, var_mapping)
                relax_mdl.add(copied_ct)
    return unrelaxables


def make_relaxed_model(mdl: Model) -> Model:
    """Return a continuous relaxation of the model.

    Variable types are set to continuous (note that semi-xxx variables have
    their LB set to zero)
    SOS var sets are ignored (could we convert them to linear cts)
    Piecewise linear cause an error

    :param mdl: the initial model

    :return: a new model with continuous relaxation, if possible, else None.
    """
    if mdl._pwl_counter:
        mdl.fatal('Model has piecewise-linear expressions, cannot be relaxed')
    mdl_class = mdl.__class__
    relaxed_model = mdl_class(name='lp_' + mdl.name)

    # transfer kwargs
    relaxed_model._parse_kwargs(mdl._get_kwargs())

    # transfer variable containers TODO
    ctn_map = {}
    for ctn in mdl.iter_var_containers():
        copied_ctn = ctn.copy_relaxed(relaxed_model)
        relaxed_model._add_var_container(copied_ctn)
        ctn_map[ctn] = copied_ctn

    var_mapping = transf_vars(mdl, relaxed_model, ctn_map)
    unrelaxables = transf_constraints(mdl, relaxed_model, var_mapping)

    # clone objective
    relaxed_model.objective_sense = mdl.objective_sense
    relaxed_model.objective_expr = mdl.objective_expr.copy(
        relaxed_model, var_mapping)

    # clone kpis
    for kpi in mdl.iter_kpis():
        relaxed_model.add_kpi(kpi.copy(relaxed_model, var_mapping))

    if mdl.context:
        relaxed_model.context = mdl.context.copy()

    # ignore sos? or convert them to additional constraints???
    for sos in mdl.iter_sos():
        if not sos.is_generated():
            unrelaxables.append(sos)

    for urx in unrelaxables:
        print('- Modeling element cannot be relaxed: {0!r}, ignored', urx)
    if unrelaxables:
        #
        return None
    else:
        # force cplex if any, on docloud nothing to do...
        cpx = relaxed_model.get_cplex(do_raise=False)
        if cpx:
            # force type to LP
            cpx.set_problem_type(0)  # 0 is code for LP.

        return relaxed_model
