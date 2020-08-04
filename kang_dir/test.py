from pyomo import environ as pe
from pyomo.opt import SolverFactory

import lpParse

m = lpParse.read("/home/kdu/lpproject/test2.lp")
print(pe.model)

opt = SolverFactory('glpk')

results = opt.solve(pe.model)
