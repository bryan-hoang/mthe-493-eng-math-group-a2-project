import time
from typing import List, Tuple

import pulp
from pulp import const as PULP

from .error import PuLPInfeasibleError
from .model import Worker


def assign_work_pulp(
    workers: List[Worker], data_set: List, beta: int, s_min: int
) -> Tuple[List[int], int]:

    k = len(workers)
    n = len(data_set)

    c = [w.c for w in workers]

    assigned_ids = []

    # TIMING start
    start = time.time_ns() // 1000

    p = pulp.LpProblem("solve_pulp_mip", PULP.LpMinimize)

    # x, the number of data elements assigned
    x = pulp.LpVariable.dicts("x", range(k), lowBound=0, upBound=n, cat=PULP.LpInteger)
    for i in range(k):
        x[i].upBound = workers[i].s_max if workers[i].s_max >= s_min else 0
    # x_b, the auxillary indicator variables
    x_b = pulp.LpVariable.dicts(
        "x_b", range(k), lowBound=0, upBound=1, cat=PULP.LpBinary
    )

    # Set objective
    p += pulp.lpSum([c[i] * x[i] for i in range(k)]), "Total cost"

    # Constraints
    # sum to n
    p += pulp.lpSum([x[i] for i in range(k)]) == n, "Sum to n"
    # at least beta workers employed
    p += pulp.lpSum([x_b[i] for i in range(k)]) >= beta, "Beta workers employed"
    # auxillary constraints: set x_bin equal to min(x, 1) for each var
    # then if x_b[i] -> x[i] >= s_min
    for i in range(k):
        p += x[i] <= n * x_b[i]
        p += x[i] >= s_min * x_b[i]

    p.solve(pulp.apis.PULP_CBC_CMD(msg=False))

    if p.status != PULP.LpStatusOptimal:
        raise PuLPInfeasibleError()

    # extract values and assign work to workers
    allocations = [v.varValue for v in x.values()]
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

