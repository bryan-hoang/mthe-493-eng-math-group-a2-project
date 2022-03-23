import datetime
import random
import sys
from math import ceil, floor
from statistics import mean
from typing import List, NamedTuple, Union

from assign import assign_work
from error import (
    AssignmentError,
    InfeasibleWorkerCapacityError,
    InsufficientCapacityError,
    InsufficientDataError,
    InsufficientWorkersError,
)
from model import InputSet, Timing, Worker
from stats import Stats
from utils import printProgressBar


class TestCase(NamedTuple):
    input_set: InputSet
    feasible: bool
    workers: List[Worker]
    stats: Union[Stats, None]
    timing: Union[Timing, None]
    error: Union[str, None]

    def __str__(self, indent=2):
        indent_str = " " * indent
        indent_str_2 = " " * (indent + 2)
        indent_str_4 = " " * (indent + 4)
        workers_str = [
            "{}{}{}".format(
                indent_str_4, w, "," if w != self.workers[-1] else ""
            )
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


def test_random(
    n_tests,
    seed=None,
    verbose=0,
    log=True,
    log_all=False,
    k_lim=100,
    s_max_lim=200,
    c_lim=20,
    s_min_factor=0.5,
    n_factor=0.1,
) -> List[TestCase]:
    """
    Generates n random tests.

    @param n_tests: number of tests to generate
    @param seed: seed for PRNG
    @param verbose: determines amount of logging to console (values: 0, 1, 2)
    @param log: Enable log file writing (summary and infeasible/erroneous distributions)
    @param log_all: If logging enabled, log all tests generated (in addition to summary/infeasible)
    @param k_lim: Upper bound on number of workers that can be generated in a test
    @param s_max_lim: Upper bound on s_max of generated workers
    @param c_lim: Upper bound on c (cost per data) of generated workers
    @param s_min_factor: The factor by which we multiply s_max_lim to determine upper bound on s_min
    @return: List of generated test cases + results
    """

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
        print(
            "Warning: s_max_lim must be int >= 1; proceesing with s_max_lim=1"
        )
    if c_lim < 1:
        c_lim = 1
        print("Warning: c_lim must be >= 1; proceesing with c_lim=1")
    if s_min_factor <= 0:
        s_min_factor = 0.5
        print(
            "Warning: s_min_factor must be > 0; proceesing with s_min_factor=0.5"
        )

    [dt, _] = datetime.datetime.now().isoformat().split(".")
    log_file_name = dt.replace(":", "_")
    log_file = log_file_name + ".log"
    raw_log_file = log_file_name + ".raw.log"

    if not log and verbose >= 0:
        print("Logging disabled")
    elif log and not log_all and verbose >= 0:
        print(
            "Logging all disabled; will only give infeasible cases + summary info"
        )

    # keep all generated InputSets
    tests: List[TestCase] = [None] * n_tests
    # keep indexes of infeasible occurrences
    infeasible = {
        ValueError.__name__: [],
        AssignmentError.__name__: [],
        InfeasibleWorkerCapacityError.__name__: [],
        InsufficientWorkersError.__name__: [],
        InsufficientCapacityError.__name__: [],
        InsufficientDataError.__name__: [],
    }

    for i_test in range(n_tests):
        if verbose >= 0:
            printProgressBar(
                i_test,
                n_tests,
                prefix="Progress:",
                suffix="Complete",
                length=80,
            )
        # workers
        k = random.randint(k_lim // 2, k_lim)
        s_max = [random.randint(0, s_max_lim) for _ in range(k)]
        s_max_avg = mean(s_max)
        c = [round(random.random() * c_lim, 2) for _ in range(k)]
        # data set
        n = random.randint(
            max(1, floor(k * s_max_avg * n_factor)), ceil(k * s_max_avg)
        )
        data_set = [i for i in range(n)]
        # params
        s_min = round(min(n // 8, s_max_lim * s_min_factor) * random.random())
        beta = random.randint(0, k // 8)

        # keep track of this test case
        input_set = InputSet(n, beta, s_min, k, s_max, c)
        stats = None
        timing = None
        error = None

        workers = [Worker(s_max[i], c[i], i) for i in range(k)]

        # run test
        if verbose >= 1:
            print("*** RESULTS {} ***".format(i_test))
            print("n: {}, k: {}".format(n, k))
        try:
            [assigned_ids, timing] = assign_work(
                workers, data_set, beta, s_min
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
            test_case = TestCase(
                input_set, feasible, workers, stats, timing, error
            )
            tests[i_test] = test_case

    feasible_tests = [x for x in enumerate(tests) if x[1].feasible]
    n_feasible = len(feasible_tests)
    feasible_pct = round(100 * n_feasible / max(n_tests, 1))
    info = [
        "test_random output log",
        "n_tests={}, seed={}, verbose={}, log={}, log_file={}".format(
            n_tests, seed, verbose, log, log_file
        ),
        "Feasible distributions found: {} ({}%)".format(
            n_feasible, feasible_pct
        ),
        "Infeasible distributions encountered:",
        "".join(
            map(
                lambda x: "{}: {}, ".format(x[0], len(x[1])),
                infeasible.items(),
            )
        ),
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
                info += [
                    templ.format(x[0], x[1].__str__(4)) for x in feasible_tests
                ]
                info.append("")
            # add infeasible tests info
            for name, l in infeasible.items():
                info.append("{}s ({}):".format(name, len(l)))
                info += map(lambda i: templ.format(i, tests[i].__str__(4)), l)
                info.append("")
            f.writelines("\n".join(list(info)))

            if verbose >= 0:
                print("Wrote to log '{}'".format(log_file))
        if log_all:
            with open(raw_log_file, "w") as f:
                templ = "TEST {}: {{\n{}\n}}"
                raw_log = [
                    templ.format(i, tests[i].__str__(2))
                    for i in range(n_tests)
                ]
                f.writelines("\n".join(list(raw_log)))

                if verbose >= 0:
                    print("Wrote raw log dump to '{}'".format(raw_log_file))

    return tests
