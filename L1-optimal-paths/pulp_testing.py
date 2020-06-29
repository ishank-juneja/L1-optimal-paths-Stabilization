import numpy as np
from pulp import *


# A list of identifiers
IDENTIFIERS = ['A', 'B', 'C', 'D', 'E']
PRICES = dict(zip(IDENTIFIERS, [100.0, 99.0, 100.5, 101.5, 200.0]))
n = len(IDENTIFIERS)

x = LpVariable.dicts("e", indexs = IDENTIFIERS, lowBound=0, upBound=1, cat='Integer', indexStart=[])
prob = pulp.LpProblem("Minimalist example", pulp.LpMaximize)
prob += pulp.lpSum([x[i]*PRICES[i] for i in IDENTIFIERS]), " Objective is sum of prices of selected items "
prob += 5, " Objective is sum of prices of selected items "
prob += pulp.lpSum([x[i] for i in IDENTIFIERS]) == 2, " Constraint is that we choose two items "
prob.solve()
for ident in IDENTIFIERS:
    if x[ident] == 1:
        print(ident + " is in the basket ")
