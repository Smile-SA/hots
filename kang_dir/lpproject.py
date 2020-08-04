from pyomo import environ as pe
from pyomo.opt import SolverFactory

a = 370
b = 420
c = 4

model = pe.ConcreteModel()
model.x = pe.Var([1, 2], domain=pe.Binary)
model.y = pe.Var([1, 2], domain=pe.Binary)
model.Objective = pe.Objective(
    expr=a * model.x[1] + b * model.x[2] + (a - b) * model.y[1] + (a + b) * model.y[2],
    sense=pe.maximize)
model.Constraint1 = pe.Constraint(
    expr=model.x[1] + model.x[2] + model.y[1] + model.y[2] <= c)

opt = SolverFactory('glpk')

results = opt.solve(model)
model.write('test_lp.lp')

#
# Print values for each variable explicitly
#
print("Print values for each variable explicitly")
for i in model.x:
    print(str(model.x[i]), model.x[i].value)
for i in model.y:
    print(str(model.y[i]), model.y[i].value)
print("")

#
# Print values for all variables
#
print("Print values for all variables")
for v in model.component_data_objects(pe.Var):
    print(str(v), v.value)
