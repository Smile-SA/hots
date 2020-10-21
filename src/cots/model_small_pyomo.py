"""
=========
cots small_pyomo
=========

One model instance, made as an example alternative to the
'model_small_cplex' example made with CPLEX.
"""

import pandas as pd

from pyomo import environ as pe


class Model:
    """
    Class with a very small minimalist use of pyomo (for translation).

    Attributes :
    - mdl : pyomo Model (Abstract) Instance (basis)
    """

    def __init__(self, df_containers: pd.DataFrame,
                 df_nodes_meta: pd.DataFrame):
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
        self.create_data(df_containers, df_nodes_meta)

        # Create the instance by feeding the model with the data
        self.instance_model = self.mdl.create_instance(self.data)

    def build_parameters(self):
        """Build all Params and Sets."""
        # number of nodes
        self.mdl.n = pe.Param(within=pe.NonNegativeIntegers)
        # number of containers
        self.mdl.c = pe.Param(within=pe.NonNegativeIntegers)
        # number of data points
        self.mdl.t = pe.Param(within=pe.NonNegativeIntegers)

        # set of nodes
        self.mdl.N = pe.Set(dimen=1)
        # set of containers
        self.mdl.C = pe.Set(dimen=1)

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

    def create_data(self, df_containers, df_nodes_meta):
        """Create data from dataframe."""
        cap = {}
        for n, n_data in df_nodes_meta.groupby(['machine_id']):
            cap.update({n: n_data['cpu'].values[0]})
        cons = {}
        df_containers.reset_index(drop=True, inplace=True)
        for key, c_data in df_containers.groupby(['container_id', 'timestamp']):
            cons.update({key: c_data['cpu'].values[0]})

        self.data = {None: {
            'n': {None: df_containers['machine_id'].nunique()},
            'c': {None: df_containers['container_id'].nunique()},
            't': {None: df_containers['timestamp'].nunique()},
            'N': {None: df_containers['machine_id'].unique().tolist()},
            'C': {None: df_containers['container_id'].unique().tolist()},
            'cap': cap,
            'T': {None: df_containers['timestamp'].unique().tolist()},
            'cons': cons,
        }}

    def write_instance(self, filename, format='lp'):
        """Write problem instance in a file."""
        if format == 'lp':
            self.instance_model.write(filename)
        elif format == 'txt':
            with open(filename, 'w') as output_file:
                self.instance_model.pprint(output_file)
        else:
            print('\n ** Error writing file : wrong output format **\n')


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


def open_nodes_(mdl):
    """Express the numbers of open nodes."""
    return sum(mdl.a[i] for i in mdl.N)
