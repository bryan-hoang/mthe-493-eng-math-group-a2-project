from collections import namedtuple
from math import floor
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
        if len(items) > self.s_max:
            raise AssignmentError(
                "Cannot assign {} items to worker with s_max {}".format(len(items), self.s_max))
        self.assigned_work = items
        self.num_assigned = len(items)
        self.cost = self.c * len(items)

    def reassign_from(self, source: "Worker", n: int, s_min: int) -> None:
        '''Reassigns n assigned elements from source to self'''
        # check that source can afford to give n
        if n > source.num_assigned - s_min:
            template = "Cannot reassign {} elements from worker {} (s_min: {}, max reassignable: {})"
            msg = template.format(n, source, s_min, max(
                0, source.num_assigned - s_min))
            raise AssignmentError(msg)
        # check that self has capacity for n items
        elif n + self.num_assigned > self.s_max:
            template = "Cannot reassign {} elements to worker {} (s_max: {}, max reassignable: {})"
            msg = template.format(n, self, self.s_max,
                                  self.s_max - self.num_assigned)
            raise AssignmentError(msg)
        # check that we're reassigning non-zero items
        elif not n > 0:
            raise ValueError("Cannot reassign {} items".format(n))
        # we know that source can give n items, and self has room to take n items
        source_items = source.assigned_work
        new_source_items = source_items[:len(source_items) - n]
        new_self_items = self.assigned_work + \
            source_items[len(source_items) - n:]
        source.assign(new_source_items)
        self.assign(new_self_items)

    def __repr__(self):
        return self.__str__() + "\n"

    def __str__(self):
        template = "Worker<id: {}, s_max: {}, c: {}, num_assigned: {}, assigned_work: {}, cost: {}>"
        return template.format(self.id, self.s_max, self.c, self.num_assigned, repr(self.assigned_work), self.cost)


InfeasibleDist = namedtuple(
    "InfeasibleDist", "feasible excess_available excess_required")


class InsufficientWorkersError(Exception):
    def __init__(self, available: int, beta: int, s_min: int) -> None:
        template = "Not enough employable workers (available: {}, required: {}). Decrease beta ({}) and/or s_min ({})"
        msg = template.format(available, beta, beta, s_min)
        super().__init__(msg)


class InsufficientDataError(Exception):
    def __init__(self, available: int, beta: int, s_min: int) -> None:
        if available == 0:
            msg = "No data provided. Must have at least 1 data element"
        else:
            template = "Not enough data to satisfy parameters (available: {}, required: {}). Decrease beta ({}) and/or s_min ({})"
            msg = template.format(available, s_min*beta, beta, s_min)
        super().__init__(msg)


class InsufficientCapacityError(Exception):
    def __init__(self, n: int, capacity: int) -> None:
        template = "Data set exceeds worker capacity (data set size: {}, capacity: {})"
        msg = template.format(n, capacity)
        super().__init__(msg)


class InfeasibleDistributionError(Exception):
    def __init__(self, n: int, s_min: int, infeasible_result: InfeasibleDist) -> None:
        n_dec = infeasible_result.excess_required - infeasible_result.excess_available
        n_inc = s_min - infeasible_result.excess_required
        template = "No feasible distribution for workers (data set size: {}, s_min: {}).\nExcess available: {}, excess required: {}.\nIncrease n by {}, decrease n by {}, or decrease s_min"
        msg = template.format(n, s_min, infeasible_result.excess_available,
                              infeasible_result.excess_required, n_dec, n_inc)
        super().__init__(msg)


class AssignmentError(Exception):
    def __init__(self, msg) -> None:
        super().__init__(msg)


def get_capacity(workers: List[Worker]) -> int:
    return sum(map(lambda w: w.s_max, workers))


def get_employable_workers(workers: List[Worker], s_min: int) -> List[Worker]:
    eligible = [w for w in workers if w.s_max > 0 and w.s_max >= s_min]
    return eligible
    # eligible.sort(key=lambda w: w.c)
    # # very important detail: we can only employ so many workers based on n and s_min
    # # this factors into which workers are actually employable, since we must exclude workers that can't meet n when in a group of max_employable
    # max_employable = n // s_min
    # # need to find the largest subset with <= max_employable elements which have sum (s_max) >= n
    # # (this is a subset-sum problem)
    # # Algorithm:
    # # start with the entire set. remove the most expensive workers until we have a subset of length max_employable
    # # if this subset satisfies sum s_max, return
    # # else, swap the worker w/ smallest s_max for the cheapest worker with a larger s_max
    # el_tuple = [(i, eligible[i].c, eligible[i].s_max) for i in range(len(eligible))]
    # incl_subset = el_tuple[:max_employable]
    # excl_subset = el_tuple[max_employable:]
    # # TODO: This could throw IndexError
    # while True:
    #     subset = [eligible[tup[0]] for tup in incl_subset]
    #     valid = get_capacity(subset)

    # return [w for w in workers if w.s_max > 0 and w.s_max >= s_min]


def check_infeasible_dist(workers: List[Worker], n: int, s_min: int):
    # Edge case: if s_min == 0, it's always feasible
    if s_min == 0:
        excess_available = sum(map(lambda w: w.s_max, workers))
        excess_required = 0
        return InfeasibleDist(True, excess_available, excess_required)
    # Assume feasible by default
    feasible = True
    # Sort workers by s_max in descending order
    sorted_workers = sorted(workers, key=lambda w: w.s_max, reverse=True)
    # upper bound determines how many workers we can employ to meet s_min requirements for n data points
    upper_bound = floor(n / s_min)
    # in the first upper_bound workers, how much excess do we have? (we can handle upper_bound * s_min to sum_{i=1}^{upper_bound}(s_max) )
    excess_available = sum([max(0, w.s_max - s_min)
                           for w in sorted_workers[:upper_bound]])
    # how many slices do we require above (s_min * upper_bound) to compute n?
    excess_required = n % s_min
    if excess_available < excess_required:
        feasible = False
    return InfeasibleDist(feasible, excess_available, excess_required)


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
    k_max = n // s_min
    # employable worker capacity = sum (s_max) over employable workers
    capacity = get_capacity(employable)
    infeasible_dist = check_infeasible_dist(employable, n, s_min)
    # CHECK 0: n > 0
    if n <= 0:
        raise InsufficientDataError(n, beta, s_min)
    # CHECK 1: n >= beta * s_min. If s_min is 0, we pretend it's 1
    elif n < beta * max(1, s_min):
        raise InsufficientDataError(n, beta, s_min)
    # CHECK 2: num employable workers >= beta
    elif k_employable < beta:
        raise InsufficientWorkersError(k_employable, beta, s_min)
    # CHECK 3: capacity >= n
    elif capacity < n:
        raise InsufficientCapacityError(n, capacity)
    # CHECK 4: infeasible distribution
    if not infeasible_dist.feasible:
        raise InfeasibleDistributionError(n, s_min, infeasible_dist)

    # TODO: Need to eliminate workers that would never be able to handle data in the feasible solution (see notes in get_employable_workers)
    # IDs of workers that receive data
    assigned_workers = []
    # index in data_set we are assigning next
    i_data = 0
    last_complete_worker_idx = -1
    while i_data < n - 1:
        worker = employable[last_complete_worker_idx + 1]
        # how many items are remaining?
        # if this is < s_min (only possible for last worker), then it will be recorded by last_complete_worker_idx
        remaining = n - i_data
        # otherwise, compute # to assign
        x = min(worker.s_max, remaining)
        # slice data_set
        items = data_set[i_data: i_data + x]
        worker.assign(items)
        # update state
        # If we were able to assign enough data to this worker to satisfy s_min
        if x >= s_min:
            last_complete_worker_idx += 1
        assigned_workers.append(worker.id)
        i_data += x

    # Number of workers that received > 0 work
    num_workers_assigned = len(assigned_workers)
    # Number of workers that received >= s_min work
    num_complete_workers_assigned = last_complete_worker_idx + 1
    # initialize the donor worker index
    donor_worker_idx = last_complete_worker_idx

    while num_complete_workers_assigned < max(num_workers_assigned, beta):
        # check to make sure we still have a worker that can donate
        if donor_worker_idx < 0:
            # TODO: There's a bug that occurs sometimes.
            # We need to reassign to eligible workers based on the max_num_employable
            # print(beta, s_min)
            # print(num_complete_workers_assigned,
            #       num_workers_assigned, donor_worker_idx)
            # print(employable)
            raise AssignmentError(
                "Unknown error occurred while attempting reassignment (ran out of workers that can donate)")
        source = employable[donor_worker_idx]
        # grab some reassignable data from the last complete worker
        n_available = source.num_assigned - s_min
        # keep reassigning from the source until it has none left to donate, and we still need to donate
        while n_available > 0 and num_complete_workers_assigned < max(num_workers_assigned, beta):
            # next worker we'll reassign stuff to
            next_worker = employable[last_complete_worker_idx + 1]
            # calculate how much we'll reassign
            # if s_min == 0, we need to assign at least 1 element. If num_assigned > s_min already, assign 0
            n_required = max(0, max(1, s_min) - next_worker.num_assigned)
            n_reassign = min(n_available, n_required)
            # perform reassignment
            next_worker.reassign_from(source, n_reassign, s_min)

            # update how much source can donate
            n_available -= n_reassign
            # add this worker to assigned_workers if it isn't already there
            if not next_worker.id in assigned_workers:
                num_workers_assigned += 1
                assigned_workers.append(next_worker.id)
            # check if next_worker is now complete
            if next_worker.num_assigned >= s_min:
                num_complete_workers_assigned += 1
                last_complete_worker_idx += 1
        # at this point, we've completed our reassignment or we need to move onto the next donor
        donor_worker_idx -= 1

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
        except (InsufficientWorkersError, InsufficientCapacityError, InfeasibleDistributionError, InsufficientDataError) as e:
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
