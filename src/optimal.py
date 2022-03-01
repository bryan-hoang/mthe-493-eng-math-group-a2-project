from typing import List, Union
from statistics import StatisticsError, variance, mean
import random


class Worker:
    '''
    Worker is just a temporary data structure, used to encapsulate the properties
    of a single worker. May or may not be included in final implementation
    '''

    id = 0
    s_max = 0
    c = 0
    num_assigned = 0
    assigned_work = []
    cost = 0

    def __init__(self, s_max: int, c: float, id=0) -> None:
        self.s_max = s_max
        self.c = c
        self.id = id

    def assign(self, items: List) -> None:
        self.assigned_work = items
        self.num_assigned = len(items)
        self.cost = self.c * len(items)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        template = "Worker<id: {}, s_max: {}, c: {}, num_assigned: {}, assigned_work: {}, cost: {}>"
        return template.format(self.id, self.s_max, self.c, self.num_assigned, repr(self.assigned_work), self.cost)


class InsufficientWorkersError(Exception):
    def __init__(self, available, beta, s_min) -> None:
        template = "Not enough employable workers (available: {}, required: {}). Increase beta ({}) and/or decrease s_min ({})"
        msg = template.format(available, beta, beta, s_min)
        super().__init__(msg)


class InsufficientCapacityError(Exception):
    def __init__(self, n, capacity) -> None:
        template = "Data set exceeds worker capacity (data set size: {}, capacity: {})"
        msg = template.format(n, capacity)
        super().__init__(msg)


def get_employable_workers(workers: List[Worker], s_min: int) -> List[Worker]:
    return [w for w in workers if w.s_max >= s_min]


def assign_work(workers: List[Worker], data_set: List, beta: int, s_min: int) -> List[int]:
    '''
    Assigns a data set to a list of workers by partitioning based on properties
    of workers (s_max, c) and parameters of the algorithm (beta, s_min)

    Returns list of worker IDs that received non-zero work
    '''

    # derive some properties from arguments
    n = len(data_set)
    k = len(workers)

    # feasibility checks
    # get employable workers (those having s_max >= s_min)
    employable = get_employable_workers(workers, s_min)
    # sort employable workers by cost (in-place)
    employable.sort(key=lambda w: w.c)
    k_employable = len(employable)
    # employable worker capacity = sum (s_max) over employable workers
    capacity = sum(map(lambda w: w.s_max, employable))
    # CHECK 1: num employable workers >= beta
    if k_employable < beta:
        raise InsufficientWorkersError(k_employable, beta, s_min)
    # TODO: Add check of n < beta * s_min before this check
    # CHECK 2: capacity >= n
    elif capacity < n:
        raise InsufficientCapacityError(n, capacity)
    # TODO: Add infeasible distribution check

    # IDs of workers that receive data
    assigned_workers = []
    # index in data_set we are assigning next
    i_data = 0
    for worker in employable:
        # how many items are remaining?
        # TODO: need to make sure we assign at least s_min to this worker
        remaining = n - i_data
        # if no items remaining, we're done
        if remaining == 0:
            break
        # otherwise, compute # to assign
        x = min(worker.s_max, remaining)
        # slice data_set
        items = data_set[i_data: i_data + x]
        worker.assign(items)
        # update state
        assigned_workers.append(worker.id)
        i_data += x

    # TODO: Implement the 'redistributing' algorithm to ensure len(assigned_workers) >= beta

    return assigned_workers


# TODO: Could move all this into a class like WorkerCollection
def get_avg_utilization(workers: List[Worker]) -> float:
    total_cap = sum(map(lambda w: w.s_max, workers))
    used_cap = sum(map(lambda w: w.num_assigned, workers))
    return used_cap / total_cap


def get_avg_c(workers: List[Worker]) -> float:
    return mean(map(lambda w: w.c, workers))


def get_avg_s_max(workers: List[Worker]) -> float:
    return mean(map(lambda w: w.s_max, workers))


def get_avg_cost(workers: List[Worker]) -> float:
    return mean(map(lambda w: w.cost, workers))


def get_avg_assigned(workers: List[Worker]) -> float:
    return mean(map(lambda w: w.num_assigned, workers))


def get_var_c(workers: List[Worker]) -> Union[float, None]:
    try:
        return variance(map(lambda w: w.c, workers))
    except StatisticsError:
        return None


def get_var_s_max(workers: List[Worker]) -> Union[float, None]:
    try:
        return variance(map(lambda w: w.s_max, workers))
    except StatisticsError:
        return None


def get_var_cost(workers: List[Worker]) -> Union[float, None]:
    try:
        return variance(map(lambda w: w.cost, workers))
    except StatisticsError:
        return None


def get_var_assigned(workers: List[Worker]) -> Union[float, None]:
    try:
        return variance(map(lambda w: w.num_assigned, workers))
    except StatisticsError:
        return None


def get_total_assigned(workers: List[Worker]) -> int:
    return sum(map(lambda w: w.num_assigned, workers))


def get_total_cost(workers: List[Worker]) -> float:
    return sum(map(lambda w: w.cost, workers))


def get_assignment_stats(workers: List[Worker], n: int, beta: int, s_min: int):
    """Computes various stats about workers"""

    stats = {}
    stats["n"] = n
    stats["beta"] = beta
    stats["s_min"] = s_min

    stats["k"] = len(workers)
    stats["total_assigned"] = get_total_assigned(workers)
    stats["total_cost"] = get_total_cost(workers)

    employable = get_employable_workers(workers, s_min)
    stats["k_employable"] = len(employable)
    stats["employable_total_ratio"] = stats["k_employable"] / stats["k"]

    assigned = [w for w in workers if w.num_assigned > 0]
    stats["k_assigned"] = len(assigned)
    stats["assigned_total_ratio"] = stats["k_assigned"] / stats["k"]
    stats["assigned_employed_ratio"] = stats["k_assigned"] / \
        stats["k_employable"]

    stats["avg_utilization"] = get_avg_utilization(workers)
    stats["avg_utilization_employable"] = get_avg_utilization(employable)
    stats["avg_utilization_assigned"] = get_avg_utilization(assigned)

    stats["avg_c"] = get_avg_c(workers)
    stats["avg_c_employable"] = get_avg_c(employable)
    stats["avg_c_assigned"] = get_avg_c(assigned)
    stats["var_c"] = get_var_c(workers)
    stats["var_c_employable"] = get_var_c(employable)
    stats["var_c_assigned"] = get_var_c(assigned)

    stats["avg_s_max"] = get_avg_s_max(workers)
    stats["avg_s_max_employable"] = get_avg_s_max(employable)
    stats["avg_s_max_assigned"] = get_avg_s_max(assigned)
    stats["var_s_max"] = get_var_s_max(workers)
    stats["var_s_max_employable"] = get_var_s_max(employable)
    stats["var_s_max_assigned"] = get_var_s_max(assigned)

    stats["avg_cost"] = get_avg_cost(workers)
    stats["avg_cost_employable"] = get_avg_cost(employable)
    stats["avg_cost_assigned"] = get_avg_cost(assigned)
    stats["var_cost"] = get_var_cost(workers)
    stats["var_cost_employable"] = get_var_cost(employable)
    stats["var_cost_assigned"] = get_var_cost(assigned)

    stats["avg_assigned"] = get_avg_assigned(workers)
    stats["avg_assigned_employable"] = get_avg_assigned(employable)
    stats["avg_assigned_assigned"] = get_avg_assigned(assigned)
    stats["var_assigned"] = get_var_assigned(workers)
    stats["var_assigned_employable"] = get_var_assigned(employable)
    stats["var_assigned_assigned"] = get_var_assigned(assigned)

    return stats


def test_random(n, seed=None, verbose=0):
    '''Generates n random tests'''

    if (seed):
        random.seed(seed)

    for i_test in range(n):
        # workers
        k = random.randint(0, 100)
        s_max = [random.randint(0, 100) for _ in range(k)]
        c = [random.randint(1, 20) for _ in range(k)]
        # data set
        n = random.randint(0, 1000)
        data_set = [i for i in range(n)]
        # params
        s_min = random.randint(0, 20)
        beta = random.randint(0, k)

        workers = [Worker(s_max[i], c[i], i) for i in range(k)]

        # run test
        print("*** RESULTS {} ***".format(i_test + 1))
        if verbose >= 1:
            print("n: {}, k: {}".format(n, k))
        try:
            assigned_ids = assign_work(workers, data_set, beta, s_min)
            assigned = [w for w in workers if w.id in assigned_ids]

            stats = get_assignment_stats(workers, n, beta, s_min)

            print("Total cost: {}, total assigned: {}".format(
                stats["total_cost"], stats["total_assigned"]))
            if verbose >= 1:
                print(stats)
            elif verbose >= 2:
                print("\nAll workers:")
                print(workers)
                print("\nAssigned workers:")
                print(assigned)
        except (InsufficientWorkersError, InsufficientCapacityError) as e:
            print(e)


if __name__ == "__main__":
    # System parameters
    # MARK: Workers
    # number of available workers in the system
    k = 5
    # upper bound estimates of workers (must be of length k)
    # we are assuming these are already provided from Duncan's piece based on benchmarks + total training time param
    s_max = [3, 6, 9, 12, 15]
    # worker costs (must be of length k)
    c = [1, 2, 3, 4, 5]

    # some checks to ensure proper lengths
    if len(s_max) != k:
        raise ValueError(
            "Invalid s_max list: expected length: {}; actual length: {}".format(k, len(s_max)))
    elif len(c) != k:
        raise ValueError(
            "Invalid c list: expected length: {}; actual length: {}".format(k, len(c)))

    # instantiate workers
    workers = [Worker(s_max[i], c[i], i) for i in range(k)]

    # MARK: data set
    n = 30
    # data set is temporarily integers
    data_set = [i for i in range(n)]

    # MARK: algorithm parameters
    # minimum number of employed workers
    beta = 3
    # minimum assignable amount of work
    s_min = 4

    assigned_ids = assign_work(workers, data_set, beta, s_min)
    assigned = [w for w in workers if w.id in assigned_ids]

    stats = get_assignment_stats(workers, n, beta, s_min)

    print("*** RESULTS ***")
    print(stats)
    print("\nAll workers:")
    print(workers)
    print("\nAssigned workers:")
    print(assigned)
