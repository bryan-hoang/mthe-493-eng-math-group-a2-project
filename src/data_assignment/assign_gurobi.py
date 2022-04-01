import time
from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

from .error import GurobiInfeasibleError
from .model import Worker


def assign_work_gurobi(
    workers: List[Worker], data_set: List, beta: int, s_min: int
) -> Tuple[List[int], int]:

    k = len(workers)
    n = len(data_set)

    c = [w.c for w in workers]

    lower_bounds = [0 for _ in range(k)]
    upper_bounds = [w.s_max if w.s_max >= s_min else 0 for w in workers]

    assigned_ids = []

    # TIMING start
    start = time.time_ns() // 1000

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()
        with gp.Model(name="solve_gurobi_mip", env=env) as m:

            # x values, the amount of data each worker is to be assigned
            x = m.addVars(k, vtype=GRB.INTEGER, lb=lower_bounds, ub=upper_bounds)
            # indicator variables, 0 or 1 if a worker is receiving work or not
            x_bin = m.addVars(k, vtype=GRB.INTEGER, lb=[0] * k, ub=[1] * k)

            # auxillary constraints: set x_bin equal to min(x, 1) for each var
            for i in range(k):
                xi = x.select(i)[0]
                xbi = x_bin.select(i)[0]
                name = "zero_{}".format(i)
                m.addGenConstrMin(xbi, [xi, 1.0], name=name)
                # indicator constraint: if x_bin is 1, then x >= s_min
                m.addConstr((xbi == 1) >> (xi >= s_min))

            # Objective function: dot product of c * x
            obj_expr = gp.LinExpr()
            obj_expr.addTerms(c, x.select())
            m.setObjective(obj_expr, GRB.MINIMIZE)

            # Constraint: x sums to n
            m.addConstr(x.sum() == n, "x_sum_n")
            m.addConstr(x_bin.sum() >= beta, "beta")

            # run solver
            m.optimize()

            # check if feasible
            if m.Status != GRB.OPTIMAL:
                raise GurobiInfeasibleError()

            # extract values and assign work to workers
            allocations = [xi.X for xi in x.select()]
            idx = 0
            for i, val in enumerate(allocations):
                w = workers[i]
                # Gurobi MIP models use floats for values that are constrained to be "nearly" integers.
                # They always need to be rounded, but are always near int values (so rounding doesn't impact solution)
                int_val = round(val)
                diff = abs(val - int_val)
                if diff > 0.001:
                    print("BIG DIFF", diff)
                w_data = data_set[idx : idx + int_val]
                w.assign(w_data)
                idx += int_val
                if len(w_data) > 0:
                    assigned_ids.append(w.id)

    duration = (time.time_ns() // 1000) - start

    return assigned_ids, duration

