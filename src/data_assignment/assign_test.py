import datetime
import random
import sys
from math import ceil, floor
from statistics import mean
import time
from typing import Dict, List, NamedTuple, Tuple, Union

import gurobipy as gp
import pulp

from .assign_pulp import assign_work_pulp
from .assign_heuristic import assign_work_heuristic
from .assign_gurobi import assign_work_gurobi
from .error import (
    AssignmentError,
    InfeasibleWorkerCapacityError,
    InsufficientCapacityError,
    InsufficientDataError,
    InsufficientWorkersError,
    GurobiInfeasibleError,
    PuLPInfeasibleError,
)
from .model import InputSet, Timing, Worker
from .stats import Stats
from .utils import printProgressBar


class TestCase(NamedTuple):
    input_set: InputSet
    feasible: bool
    workers: List[Worker]
    stats: Union[Stats, None]
    timing: Union[Timing, float, None]
    error: Union[str, None]

    def __str__(self, indent=2):
        indent_str = " " * indent
        indent_str_2 = " " * (indent + 2)
        indent_str_4 = " " * (indent + 4)
        workers_str = [
            "{}{}{}".format(indent_str_4, w, "," if w != self.workers[-1] else "")
            for w in self.workers
        ]
        workers_str = "\n".join(workers_str)

        str_val = "{}TestCase(\n".format(indent_str)
        str_val += "{}input_set={}\n".format(indent_str_2, self.input_set)
        str_val += "{}workers={{\n{}\n{}}}\n".format(
            indent_str_2, workers_str, indent_str_2
        )
        str_val += "{}feasible={}\n".format(indent_str_2, self.feasible)
        str_val += "{}stats={}\n".format(indent_str_2, self.stats)
        str_val += "{}timing={}\n".format(indent_str_2, self.timing)
        str_val += "{}error={}\n".format(indent_str_2, self.error)
        str_val += "{})".format(indent_str)

        return str_val


class InstantiatedInputSet(NamedTuple):
    input_set: InputSet
    workers: List[Worker]
    data_set: List


def gen_test_inputs(
    n_tests,
    seed=None,
    k_lim=100,
    s_max_lim=200,
    c_lim=20,
    s_min_factor=0.5,
    n_factor=0.1,
) -> List[InstantiatedInputSet]:
    if seed:
        random.seed(seed)
    else:
        seed = random.randrange(sys.maxsize)
        random.seed(seed)
    if k_lim < 1 or type(k_lim) != int:
        k_lim = 1
        print("Warning: k_lim must be int >= 1; proceesing with k_lim=1")
    if s_max_lim < 1 or type(s_max_lim) != int:
        s_max_lim = 1
        print("Warning: s_max_lim must be int >= 1; proceesing with s_max_lim=1")
    if c_lim < 1:
        c_lim = 1
        print("Warning: c_lim must be >= 1; proceesing with c_lim=1")
    if s_min_factor <= 0:
        s_min_factor = 0.5
        print("Warning: s_min_factor must be > 0; proceesing with s_min_factor=0.5")

    # keep all generated InputSets
    generated: List[InstantiatedInputSet] = [None] * n_tests

    for i_test in range(n_tests):
        # workers
        k = random.randint(k_lim // 2, k_lim)
        s_max = [random.randint(0, s_max_lim) for _ in range(k)]
        s_max_avg = mean(s_max)
        c = [round(random.random() * c_lim, 2) for _ in range(k)]
        # data set
        n = random.randint(max(1, floor(k * s_max_avg * n_factor)), ceil(k * s_max_avg))
        data_set = [i for i in range(n)]
        # params
        s_min = round(min(n // 8, s_max_lim * s_min_factor) * random.random())
        beta = random.randint(0, k // 8)

        input_set = InputSet(n, beta, s_min, k, s_max, c)
        workers = [Worker(s_max[i], c[i], i) for i in range(k)]

        x = InstantiatedInputSet(input_set, workers, data_set)

        generated[i_test] = x

    return generated


def log_test_output(
    algo: str,
    test_output: List[TestCase],
    infeasible: Dict[str, List[int]],
    duration: int,
    verbose=0,
    log=True,
    log_all=False,
):
    n_tests = len(test_output)

    [dt, _] = datetime.datetime.now().isoformat().split(".")
    log_file_timestamp = dt.replace(":", "_")
    log_file = "{}-{}.log".format(algo, log_file_timestamp)
    raw_log_file = "{}-{}.raw.log".format(algo, log_file_timestamp)

    feasible_tests = [x for x in enumerate(test_output) if x[1].feasible]
    n_feasible = len(feasible_tests)
    feasible_pct = round(100 * n_feasible / max(n_tests, 1))
    info = [
        "test_random output log",
        "n_tests={}, verbose={}, log={}, log_file={}".format(
            n_tests, verbose, log, log_file
        ),
        "TOTAL DURATION (ms): {}".format(duration),
        "Feasible distributions found: {} ({}%)".format(n_feasible, feasible_pct),
        "Infeasible distributions encountered:",
        "".join(map(lambda x: "{}: {}, ".format(x[0], len(x[1])), infeasible.items(),)),
        "",
    ]
    if verbose >= 0:
        print("\nTests complete")
        print("\n".join(list(info)))

    if log:
        if verbose >= 0:
            print("Writing logs...")
        with open(log_file, "w") as f:
            templ = "  TEST {}: {{\n{}\n  }}"
            if log_all:
                # add feasible tests
                info.append("Feasible Tests ({}):".format(n_feasible))
                info += [templ.format(x[0], x[1].__str__(4)) for x in feasible_tests]
                info.append("")
            # add infeasible tests info
            for name, l in infeasible.items():
                info.append("{}s ({}):".format(name, len(l)))
                info += map(lambda i: templ.format(i, test_output[i].__str__(4)), l)
                info.append("")
            f.writelines("\n".join(list(info)))

            if verbose >= 0:
                print("Wrote to log '{}'".format(log_file))
        if log_all:
            with open(raw_log_file, "w") as f:
                templ = "TEST {}: {{\n{}\n}}"
                raw_log = [
                    templ.format(i, test_output[i].__str__(2)) for i in range(n_tests)
                ]
                f.writelines("\n".join(list(raw_log)))

                if verbose >= 0:
                    print("Wrote raw log dump to '{}'".format(raw_log_file))


def run_tests_heuristic(
    tests: List[InstantiatedInputSet], verbose=0, log=True, log_all=False,
) -> Tuple[List[TestCase], int]:
    """
    Runs the list of tests specified under heuristic allocation algorithm (assign_work)

    @param tests: tests to run
    @param verbose: determines amount of logging to console (values: 0, 1, 2)
    @param log: Enable log file writing (summary and infeasible/erroneous distributions)
    @param log_all: If logging enabled, log all tests generated (in addition to summary/infeasible)
    """

    if not log and verbose >= 0:
        print("Logging disabled")
    elif log and not log_all and verbose >= 0:
        print("Logging all disabled; will only give infeasible cases + summary info")

    # keep indexes of infeasible occurrences
    infeasible = {
        ValueError.__name__: [],
        AssignmentError.__name__: [],
        InfeasibleWorkerCapacityError.__name__: [],
        InsufficientWorkersError.__name__: [],
        InsufficientCapacityError.__name__: [],
        InsufficientDataError.__name__: [],
    }

    n_tests = len(tests)
    test_output: List[TestCase] = [None] * n_tests

    # TIME start
    start = time.time_ns() // 1000

    for i_test in range(n_tests):
        if verbose >= 0:
            printProgressBar(
                i_test, n_tests, prefix="Progress:", suffix="Complete", length=80,
            )
        input_set, workers, data_set = tests[i_test]

        stats = None
        timing = None
        error = None

        # run test
        if verbose >= 1:
            print("*** RESULTS {} ***".format(i_test))
            print("n: {}, k: {}".format(input_set.n, input_set.k))
        try:
            [assigned_ids, timing] = assign_work_heuristic(
                workers, data_set, input_set.beta, input_set.s_min
            )
            assigned = [w for w in workers if w.id in assigned_ids]

            stats = Stats(input_set, workers)

            if verbose >= 1:
                print(stats)
            elif verbose >= 2:
                print("\nAll workers:")
                print(workers)
                print("\nAssigned workers:")
                print(assigned)
        except (
            InsufficientWorkersError,
            InsufficientCapacityError,
            InsufficientDataError,
            InfeasibleWorkerCapacityError,
            AssignmentError,
            ValueError,
        ) as e:
            infeasible[e.__class__.__name__].append(i_test)
            error = repr(e)
            if verbose >= 1:
                print(e)
        finally:
            feasible = error is None
            test_case = TestCase(input_set, feasible, workers, stats, timing, error)
            test_output[i_test] = test_case

    # TIME end
    duration = (time.time_ns() // 1000) - start

    log_test_output(
        "heuristic", test_output, infeasible, duration, verbose, log, log_all
    )

    return test_output, duration


def run_tests_gurobi(
    tests: List[InstantiatedInputSet], verbose=0, log=True, log_all=False
) -> Tuple[List[TestCase], int]:
    """
    Runs the list of tests specified under gurobi allocation algorithm (assign_work_gurobi)

    @param tests: tests to run
    @param verbose: determines amount of logging to console (values: 0, 1, 2)
    @param log: Enable log file writing (summary and infeasible/erroneous distributions)
    @param log_all: If logging enabled, log all tests generated (in addition to summary/infeasible)
    """

    if not log and verbose >= 0:
        print("Logging disabled")
    elif log and not log_all and verbose >= 0:
        print("Logging all disabled; will only give infeasible cases + summary info")

    n_tests = len(tests)
    test_output: List[TestCase] = [None] * n_tests

    # Infeasible tests indices
    infeasible = {
        GurobiInfeasibleError.__name__: [],
        gp.GurobiError.__name__: [],
    }

    # TIME start
    start = time.time_ns() // 1000

    for i_test in range(n_tests):
        if verbose >= 0:
            printProgressBar(
                i_test, n_tests, prefix="Progress:", suffix="Complete", length=80,
            )
        input_set, workers, data_set = tests[i_test]

        stats = None
        timing = None
        error = None

        # run test
        if verbose >= 1:
            print("*** RESULTS {} ***".format(i_test))
            print("n: {}, k: {}".format(input_set.n, input_set.k))
        try:
            [assigned_ids, timing] = assign_work_gurobi(
                workers, data_set, input_set.beta, input_set.s_min
            )
            assigned = [w for w in workers if w.id in assigned_ids]

            stats = Stats(input_set, workers)

            if verbose >= 1:
                print(stats)
            elif verbose >= 2:
                print("\nAll workers:")
                print(workers)
                print("\nAssigned workers:")
                print(assigned)
        except (gp.GurobiError, GurobiInfeasibleError) as e:
            infeasible[e.__class__.__name__].append(i_test)
            error = repr(e)
            if verbose >= 1:
                print(e)
        finally:
            feasible = error is None
            test_case = TestCase(input_set, feasible, workers, stats, timing, error)
            test_output[i_test] = test_case

    # TIME end
    duration = (time.time_ns() // 1000) - start

    log_test_output("gurobi", test_output, infeasible, duration, verbose, log, log_all)

    return test_output, duration


def run_tests_pulp(
    tests: List[InstantiatedInputSet], verbose=0, log=True, log_all=False
) -> Tuple[List[TestCase], int]:
    """
    Runs the list of tests specified under pulp allocation algorithm (assign_work_pulp)

    @param tests: tests to run
    @param verbose: determines amount of logging to console (values: 0, 1, 2)
    @param log: Enable log file writing (summary and infeasible/erroneous distributions)
    @param log_all: If logging enabled, log all tests generated (in addition to summary/infeasible)
    """

    if not log and verbose >= 0:
        print("Logging disabled")
    elif log and not log_all and verbose >= 0:
        print("Logging all disabled; will only give infeasible cases + summary info")

    n_tests = len(tests)
    test_output: List[TestCase] = [None] * n_tests

    # Infeasible tests indices
    infeasible = {
        PuLPInfeasibleError.__name__: [],
        pulp.const.PulpError.__name__: [],
    }

    # TIME start
    start = time.time_ns() // 1000

    for i_test in range(n_tests):
        if verbose >= 0:
            printProgressBar(
                i_test, n_tests, prefix="Progress:", suffix="Complete", length=80,
            )
        input_set, workers, data_set = tests[i_test]

        stats = None
        timing = None
        error = None

        # run test
        if verbose >= 1:
            print("*** RESULTS {} ***".format(i_test))
            print("n: {}, k: {}".format(input_set.n, input_set.k))
        try:
            [assigned_ids, timing] = assign_work_pulp(
                workers, data_set, input_set.beta, input_set.s_min
            )
            assigned = [w for w in workers if w.id in assigned_ids]

            stats = Stats(input_set, workers)

            if verbose >= 1:
                print(stats)
            elif verbose >= 2:
                print("\nAll workers:")
                print(workers)
                print("\nAssigned workers:")
                print(assigned)
        except (pulp.const.PulpError, PuLPInfeasibleError) as e:
            infeasible[e.__class__.__name__].append(i_test)
            error = repr(e)
            if verbose >= 1:
                print(e)
        finally:
            feasible = error is None
            test_case = TestCase(input_set, feasible, workers, stats, timing, error)
            test_output[i_test] = test_case

    # TIME end
    duration = (time.time_ns() // 1000) - start

    log_test_output("pulp", test_output, infeasible, duration, verbose, log, log_all)

    return test_output, duration


def validate(
    n, run_a, run_b, seed=1, verbose=0, log=True, log_all=False, del_after=True
):
    # tolerance for cost difference (gurobi has rounding errors)
    TOLERANCE = 0.001
    # gen tests
    print("*** Generating tests ***")
    tests_a = gen_test_inputs(n, seed)
    tests_b = gen_test_inputs(n, seed)

    # run
    print("\n*** Running method A ***")
    output_a, duration_a = run_a(tests_a, verbose, log, log_all)
    print("\n*** Running method B ***")
    output_b, duration_b = run_b(tests_b, verbose, log, log_all)

    comparison = []
    issues = []

    n_a_suboptimal = 0
    n_b_suboptimal = 0
    n_feasibility_mismatch = 0

    # validate output
    print("\n*** Validating solutions ***")
    for i, (test_a, test_b) in enumerate(zip(output_a, output_b)):
        # A output
        feasible_a = test_a.feasible
        stats_a = test_a.stats
        # B output
        feasible_b = test_b.feasible
        stats_b = test_b.stats

        if (not feasible_a) and (not feasible_b):
            msg = "({}) both feasible".format(i)
            comparison.append(msg)
        elif feasible_a != feasible_b:
            msg = "({}) Feasibility mismatch: A: {}, B: {}".format(
                i, feasible_a, feasible_b
            )
            comparison.append(msg)
            issues.append(msg)
            n_feasibility_mismatch += 1
        else:
            # compare costs
            cost_a = stats_a.total_cost
            cost_b = stats_b.total_cost
            diff = abs(cost_a - cost_b)
            if diff > TOLERANCE:
                msg = "({}) Cost mismatch: diff: {} (A: {}, B: {})".format(
                    i, diff, cost_a, cost_b
                )
                comparison.append(msg)
                issues.append(msg)
                if cost_a > cost_b:
                    n_a_suboptimal += 1
                else:
                    n_b_suboptimal += 1
            else:
                msg = "({}) Same cost".format(i)
                comparison.append(msg)

    if del_after:
        del tests_a, tests_b, output_a, output_b

    n_suboptimal = n_a_suboptimal + n_b_suboptimal

    print("\n*** Stats ***")
    print("Tests generated:", n)
    print(
        "Feasibility mismatches: {} ({:.2%})".format(
            n_feasibility_mismatch, n_feasibility_mismatch / n
        )
    )
    print("Optimality mismatches: {} ({:.2%})".format(n_suboptimal, n_suboptimal / n))
    print(
        "A runtime: {}ms (total), {:0.2f}ms (mean)".format(duration_a, duration_a / n)
    )
    print("A suboptimality: {} ({:.2%})".format(n_a_suboptimal, n_a_suboptimal / n))
    print(
        "B runtime: {}ms (total), {:0.2f}ms (mean)".format(duration_b, duration_b / n)
    )
    print("B suboptimality: {} ({:.2%})".format(n_b_suboptimal, n_b_suboptimal / n))

    return comparison, issues

