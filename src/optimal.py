from collections import namedtuple
from math import ceil, floor
from re import template
import sys
from typing import List, Union, NamedTuple, Tuple
from statistics import StatisticsError, variance, mean
import random
from itertools import combinations
import time
import datetime

# Sets whether we use subset brute-force method or optimized method (WIP)
USE_SUBSET = True
# Sets whether a print of Worker will display the data actually allocated
LOG_ASSIGNED_WORK = False
# Should InputSet(c=[...], s_max=[...]) be logged?
LOG_INPUT_LISTS = True
# Should the timing in assign_work be enabled?
ENABLE_NATIVE_TIMING = False


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
        return self.__str__()

    def __str__(self):
        if LOG_ASSIGNED_WORK:
            template = "Worker(id: {}, s_max: {}, c: {}, num_assigned: {}, assigned_work: {}, cost: {})"
            return template.format(self.id, self.s_max, self.c, self.num_assigned, repr(self.assigned_work), round(self.cost, 2))
        else:
            template = "Worker(id: {}, s_max: {}, c: {}, num_assigned: {}, cost: {})"
            return template.format(self.id, self.s_max, self.c, self.num_assigned, round(self.cost, 2))


class InputSet(NamedTuple):
    n: int
    beta: int
    s_min: int
    k: int
    s_max: List[int]
    c: List[Union[int, float]]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        log_keys = ["n", "beta", "s_min", "k"]
        if LOG_INPUT_LISTS:
            log_keys += ["s_max", "c"]
        str_val = "InputSet("
        for i in range(len(log_keys)):
            is_last = i == len(log_keys) - 1
            str_val += "{}={}{}".format(log_keys[i],
                                        self[i], ", " if not is_last else "")
        str_val += ")"
        return str_val


class Timing(NamedTuple):
    init: float
    get_employable_workers: float
    get_capacity: float
    assign: float
    reassign: float
    total: float


class Stats:

    stat_list = ["n", "beta", "s_min", "k", "total_assigned", "total_cost", "k_employable", "max_employable", "employable_total_ratio", "k_assigned", "assigned_total_ratio", "assigned_employed_ratio", "avg_utilization", "avg_utilization_employable", "avg_utilization_assigned", "avg_c", "avg_c_employable", "avg_c_assigned", "var_c", "var_c_employable", "var_c_assigned",
                 "avg_s_max", "avg_s_max_employable", "avg_s_max_assigned", "var_s_max", "var_s_max_employable", "var_s_max_assigned", "avg_cost", "avg_cost_employable", "avg_cost_assigned", "var_cost", "var_cost_employable", "var_cost_assigned", "avg_assigned", "avg_assigned_employable", "avg_assigned_assigned", "var_assigned", "var_assigned_employable", "var_assigned_assigned"]
    stats_pct_list = ["employable_total_ratio", "assigned_total_ratio", "assigned_employed_ratio",
                      "avg_utilization", "avg_utilization_employable", "avg_utilization_assigned"]

    input_set: InputSet
    workers: List[Worker]
    employable: List[Worker]
    assigned: List[Worker]

    n: int
    beta: int
    s_min: int
    k: int
    total_assigned: int
    total_cost: float
    k_employable: int
    max_employable: int
    employable_total_ratio: float
    k_assigned: int
    assigned_total_ratio: float
    assigned_employed_ratio: float
    avg_utilization: float
    avg_utilization_employable: float
    avg_utilization_assigned: float
    avg_c: float
    avg_c_employable: float
    avg_c_assigned: float
    var_c: float
    var_c_employable: float
    var_c_assigned: float
    avg_s_max: float
    avg_s_max_employable: float
    avg_s_max_assigned: float
    var_s_max: float
    var_s_max_employable: float
    var_s_max_assigned: float
    avg_cost: float
    avg_cost_employable: float
    avg_cost_assigned: float
    var_cost: float
    var_cost_employable: float
    var_cost_assigned: float
    avg_assigned: float
    avg_assigned_employable: float
    avg_assigned_assigned: float
    var_assigned: float
    var_assigned_employable: float
    var_assigned_assigned: float

    def __init__(self, input_set: InputSet, workers: List[Worker], employable=None):
        self.input_set = input_set
        self.workers = workers
        self.employable = employable or get_employable_workers(
            workers, input_set.s_min, input_set.n)
        self.assigned = [w for w in workers if w.num_assigned > 0]
        self.calc_stats()

    @staticmethod
    def get_avg_utilization(workers: List[Worker]) -> Union[float, None]:
        total_cap = sum(map(lambda w: w.s_max, workers))
        used_cap = sum(map(lambda w: w.num_assigned, workers))
        try:
            return used_cap / total_cap
        except ZeroDivisionError:
            return None

    @staticmethod
    def get_avg_c(workers: List[Worker]) -> Union[float, None]:
        try:
            return mean(map(lambda w: w.c, workers))
        except StatisticsError:
            return None

    @staticmethod
    def get_avg_s_max(workers: List[Worker]) -> Union[float, None]:
        try:
            return mean(map(lambda w: w.s_max, workers))
        except StatisticsError:
            return None

    @staticmethod
    def get_avg_cost(workers: List[Worker]) -> Union[float, None]:
        try:
            return mean(map(lambda w: w.cost, workers))
        except StatisticsError:
            return None

    @staticmethod
    def get_avg_assigned(workers: List[Worker]) -> Union[float, None]:
        try:
            return mean(map(lambda w: w.num_assigned, workers))
        except StatisticsError:
            return None

    @staticmethod
    def get_var_c(workers: List[Worker]) -> Union[float, None]:
        try:
            return variance(map(lambda w: w.c, workers))
        except StatisticsError:
            return None

    @staticmethod
    def get_var_s_max(workers: List[Worker]) -> Union[float, None]:
        try:
            return variance(map(lambda w: w.s_max, workers))
        except StatisticsError:
            return None

    @staticmethod
    def get_var_cost(workers: List[Worker]) -> Union[float, None]:
        try:
            return variance(map(lambda w: w.cost, workers))
        except StatisticsError:
            return None

    @staticmethod
    def get_var_assigned(workers: List[Worker]) -> Union[float, None]:
        try:
            return variance(map(lambda w: w.num_assigned, workers))
        except StatisticsError:
            return None

    @staticmethod
    def get_total_assigned(workers: List[Worker]) -> int:
        return sum(map(lambda w: w.num_assigned, workers))

    @staticmethod
    def get_total_cost(workers: List[Worker]) -> float:
        return sum(map(lambda w: w.cost, workers))

    def calc_stats(self):
        self.n = self.input_set.n
        self.beta = self.input_set.beta
        self.s_min = self.input_set.s_min

        self.k = len(self.workers)
        self.total_assigned = Stats.get_total_assigned(self.workers)
        self.total_cost = Stats.get_total_cost(self.workers)

        self.k_employable = len(self.employable)
        self.max_employable = self.n // self.s_min if self.s_min > 0 else self.k_employable
        self.employable_total_ratio = self.k_employable / self.k

        self.k_assigned = len(self.assigned)
        self.assigned_total_ratio = self.k_assigned / self.k
        self.assigned_employed_ratio = self.k_assigned / self.k_employable

        self.avg_utilization = Stats.get_avg_utilization(self.workers)
        self.avg_utilization_employable = Stats.get_avg_utilization(
            self.employable)
        self.avg_utilization_assigned = Stats.get_avg_utilization(
            self.assigned)

        self.avg_c = Stats.get_avg_c(self.workers)
        self.avg_c_employable = Stats.get_avg_c(self.employable)
        self.avg_c_assigned = Stats.get_avg_c(self.assigned)
        self.var_c = Stats.get_var_c(self.workers)
        self.var_c_employable = Stats.get_var_c(self.employable)
        self.var_c_assigned = Stats.get_var_c(self.assigned)

        self.avg_s_max = Stats.get_avg_s_max(self.workers)
        self.avg_s_max_employable = Stats.get_avg_s_max(self.employable)
        self.avg_s_max_assigned = Stats.get_avg_s_max(self.assigned)
        self.var_s_max = Stats.get_var_s_max(self.workers)
        self.var_s_max_employable = Stats.get_var_s_max(self.employable)
        self.var_s_max_assigned = Stats.get_var_s_max(self.assigned)

        self.avg_cost = Stats.get_avg_cost(self.workers)
        self.avg_cost_employable = Stats.get_avg_cost(self.employable)
        self.avg_cost_assigned = Stats.get_avg_cost(self.assigned)
        self.var_cost = Stats.get_var_cost(self.workers)
        self.var_cost_employable = Stats.get_var_cost(self.employable)
        self.var_cost_assigned = Stats.get_var_cost(self.assigned)

        self.avg_assigned = Stats.get_avg_assigned(self.workers)
        self.avg_assigned_employable = Stats.get_avg_assigned(self.employable)
        self.avg_assigned_assigned = Stats.get_avg_assigned(self.assigned)
        self.var_assigned = Stats.get_var_assigned(self.workers)
        self.var_assigned_employable = Stats.get_var_assigned(self.employable)
        self.var_assigned_assigned = Stats.get_var_assigned(self.assigned)

    def __str__(self):
        values = vars(self)
        str_val = "Stats("
        for key in Stats.stat_list:
            val = values[key]
            if val is None:
                pass
            elif key in Stats.stats_pct_list:
                key += "_pct"
                val = round(val * 100, 2)
            elif type(val) == float:
                val = round(val, 1)
            str_val += "{}: {}, ".format(key, val)
        str_val += ")"
        return str_val


class TestCase(NamedTuple):
    input_set: InputSet
    feasible: bool
    workers: List[Worker]
    stats: Union[Stats, None]
    timing: Union[Timing, None]
    error: Union[str, None]

    def __str__(self, indent=2):
        indent_str = " "*indent
        indent_str_2 = " "*(indent + 2)
        indent_str_4 = " "*(indent + 4)
        workers_str = ["{}{}{}".format(indent_str_4,
                                       w, "," if w != self.workers[-1] else "") for w in self.workers]
        workers_str = "\n".join(workers_str)

        str_val = "{}TestCase(\n".format(indent_str)
        str_val += "{}input_set={}\n".format(indent_str_2, self.input_set)
        str_val += "{}workers={{\n{}\n{}}}\n".format(
            indent_str_2, workers_str, indent_str_2)
        str_val += "{}feasible={}\n".format(indent_str_2, self.feasible)
        str_val += "{}stats={}\n".format(indent_str_2, self.stats)
        str_val += "{}timing={}\n".format(indent_str_2, self.timing)
        str_val += "{}error={}\n".format(indent_str_2, self.error)
        str_val += "{})".format(indent_str)

        return str_val


class InsufficientWorkersError(Exception):
    __name__ = "InsufficientWorkersError"

    def __init__(self, available: int, beta: int, s_min: int) -> None:
        template = "Not enough employable workers (available: {}, required: {}). Decrease beta ({}) and/or s_min ({})"
        msg = template.format(available, beta, beta, s_min)
        super().__init__(msg)


class InsufficientDataError(Exception):
    __name__ = "InsufficientDataError"

    def __init__(self, available: int, beta: int, s_min: int) -> None:
        template = "Not enough data to satisfy parameters (available: {}, required: {}). Decrease beta ({}) and/or s_min ({})"
        msg = template.format(available, max(1, s_min)*beta, beta, s_min)
        super().__init__(msg)


class InsufficientCapacityError(Exception):
    __name__ = "InsufficientCapacityError"

    def __init__(self, n: int, capacity: int) -> None:
        template = "Data set size exceeds employable worker capacity (data set size: {}, capacity: {})"
        msg = template.format(n, capacity)
        super().__init__(msg)


class InfeasibleWorkerCapacityError(Exception):
    __name__ = "InfeasibleWorkerCapacityError"

    def __init__(self, k: int, n: int, s_min: int) -> None:
        max_employable = n // s_min if s_min > 0 else k
        template = "No subset of {} workers can compute data (data set size: {}, s_min: {})"
        msg = template.format(max_employable, n, s_min)
        super().__init__(msg)


class AssignmentError(Exception):
    __name__ = "AssignmentError"

    def __init__(self, msg) -> None:
        super().__init__(msg)


def get_capacity(workers: List[Worker]) -> int:
    return sum(map(lambda w: w.s_max, workers))


# this is O(n), could be improved
def insert_to_sorted_list(l: List, x, key=None):
    if not key:
        def key(x): return x
    x_val = key(x)
    for i in range(len(l)):
        if key(l[i]) > x_val:
            break
    l.insert(i, x)


def get_employable_workers_wip(workers: List[Worker], s_min: int, n: int) -> List[Worker]:
    """This is a work-in-progress, which hopefully will be able to be optimized more than the subset brute-force method"""
    eligible = [w for w in workers if w.s_max > 0 and w.s_max >= s_min]
    # return eligible
    eligible.sort(key=lambda w: w.c)
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

    '''
    New idea: calculate the max_employable
    then, find the max_employable workers that have sum s_max >= n.
    This may be O(n!), unless we can optimize further
    (it will be a subset-sum algorithm)
    algorithm:
    1. select the cheapest max_employable workers
    2: if max_employable > k, return selected
    3. set swap_idx = len(selected) - 1 (the most expensive worker)
    4. if sum (selected s_max < n), swap the most expensive selected with the least expensive worker having larger s_max


    other thought:
    - calculate sum s_max for all subsets of max_employable workers
    '''
    max_employable = n // s_min
    # it's important that these lists remain sorted by c
    selected = eligible[:max_employable]
    unselected = eligible[max_employable:]
    # check if we've already got the best selection
    if max_employable > len(selected) or get_capacity(selected) >= n:
        return selected
    # otherwise, we need to do some swapping, starting with the most expensive selected workers
    # this doesn't guarantee an optimal solution; we will (on each pass) swap the most expensive

    # WE SHOULD EXAMINE: how much capacity do we gain vs. how much does it cost?
    # compare each selected worker to each unselected worker. calculate, for each pair, the capacity gained, and the added expense.
    # then, check if there is any single swap that will meet required capacity
    #  -> if one such swap exists, filter all swaps by those that will meet requirement. Then find cheapest one and perform it
    #  -> otherwise, we need to determine which swaps to consider, and follow multiple paths. Could:
    #     -> simulate one swap for each selected? (breadth first)
    #     -> simulate taking swap which increases capacity by the most, or swap which increases cost by the least?
    for swap_idx in range(len(selected) - 1, -1, -1):
        did_swap = False
        # pop swappable worker
        to_unselect = selected.pop(swap_idx)
        # capacity of remaining selected workers
        partial_capacity = get_capacity(selected)
        required_s_max = n - partial_capacity
        # find an unselected worker with required s_max
        for w in unselected:
            if w.s_max >= required_s_max:
                insert_to_sorted_list(selected, w, key=lambda x: x.c)
                did_swap = True
                break
        # if we didn't swap, add to_unselect back to selected
        if not did_swap:
            selected.insert(swap_idx, to_unselect)


def get_employable_workers_subset(workers: List[Worker], s_min: int, n: int) -> List[Worker]:
    """
    This version uses combinatorics to brute-force, considering sum (s_max) over all subsets of length max_employable.
    Returns subset of length min(len(eligible), k_employable) with lowest c.

    TODO: need to verify that this method does return a subset which minimizes cost
    """

    # Eligible workers are those having s_max >= 1 and s_max >= s_min
    eligible = [w for w in workers if w.s_max >= max(1, s_min)]
    # Sort by c rate, ascending
    eligible.sort(key=lambda w: w.c)
    k_eligible = len(eligible)
    max_employable = n // s_min if s_min > 0 else k_eligible
    # if we already have fewer than max_employable workers
    if k_eligible <= max_employable:
        return eligible

    # otherwise, need to consider all subsets
    # (this will be brute-force, but we could use a more sophisticated branch-and-bound method)
    subsets = combinations(eligible, max_employable)
    # since combinations follows lexicographical order of eligible, the first subset of capacity n will be the cheapest
    # NOTE: prove this, cuz im not sure that's always true (depends on how our algo determines allocation of data)
    for subset in subsets:
        if get_capacity(subset) >= n:
            return subset
    # if no subsets meet required capacity, throw an error
    raise InfeasibleWorkerCapacityError(k_eligible, n, s_min)


get_employable_workers = get_employable_workers_subset if USE_SUBSET else get_employable_workers_wip


def assign_work(workers: List[Worker], data_set: List, beta: int, s_min: int) -> Tuple[List[int], Timing]:
    '''
    Assigns a data set to a list of workers by partitioning based on properties
    of workers (s_max, c) and parameters of the algorithm (beta, s_min)

    Returns tuple: 
      [0]: list of worker IDs that received non-zero work
      [1]: timing/duration info
    '''
    # TIMING start
    duration = Timing(0, 0, 0, 0, 0, 0)
    if ENABLE_NATIVE_TIMING:
        timing = [round(time.time_ns() / 1000)]

    # derive some properties from arguments
    n = len(data_set)
    k = len(workers)

    # CHECK 0: n > 0
    if n <= 0:
        raise ValueError(
            "Must have at least one data element (n: {})".format(n))
    elif k <= 0:
        raise ValueError("Must have at least one worker (k: {})".format(k))
    # CHECK 1: n >= beta * s_min. If s_min is 0, we pretend it's 1
    elif n < beta * max(1, s_min):
        raise InsufficientDataError(n, beta, s_min)

    # TIMING init
    if ENABLE_NATIVE_TIMING:
        timing.append(round(time.time_ns() / 1000))
        duration["init"] = timing[-1] - timing[-2]

    # get employable workers (those having s_max >= s_min)
    employable = get_employable_workers(workers, s_min, n)

    # TIMING get_employable_workers
    if ENABLE_NATIVE_TIMING:
        timing.append(round(time.time_ns() / 1000))
        duration["get_employable_workers"] = timing[-1] - timing[-2]

    k_employable = len(employable)
    # employable worker capacity = sum (s_max) over employable workers
    capacity = get_capacity(employable)

    # TIMING get_capacity
    if ENABLE_NATIVE_TIMING:
        timing.append(round(time.time_ns() / 1000))
        duration["get_capacity"] = timing[-1] - timing[-2]

    # CHECK 2: num employable workers >= beta
    if k_employable < beta:
        raise InsufficientWorkersError(k_employable, beta, s_min)
    # CHECK 3: capacity >= n
    elif capacity < n:
        raise InsufficientCapacityError(n, capacity)

    # IDs of workers that receive data
    assigned_workers = []
    # index in data_set we are assigning next
    i_data = 0
    last_complete_worker_idx = -1
    while i_data < n:
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

    # TIMING assign
    if ENABLE_NATIVE_TIMING:
        timing.append(round(time.time_ns() / 1000))
        duration["assign"] = timing[-1] - timing[-2]

    # Number of workers that received > 0 work
    num_workers_assigned = len(assigned_workers)
    # Number of workers that received >= s_min work
    num_complete_workers_assigned = last_complete_worker_idx + 1
    # initialize the donor worker index
    donor_worker_idx = last_complete_worker_idx

    while num_complete_workers_assigned < max(num_workers_assigned, beta):
        # check to make sure we still have a worker that can donate
        if donor_worker_idx < 0:
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

    # TIMING reassign/end
    if ENABLE_NATIVE_TIMING:
        timing.append(round(time.time_ns() / 1000))
        duration["reassign"] = timing[-1] - timing[-2]
        duration["total"] = timing[-1] - timing[0]

    return assigned_workers, duration

# Print iterations progress


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def test_random(n_tests, seed=None, verbose=0, log=True, log_all=False, k_lim=100, s_max_lim=200, c_lim=20, s_min_factor=0.5, n_factor=0.1) -> List[TestCase]:
    '''
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
    '''

    if (seed):
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

    [dt, _] = datetime.datetime.now().isoformat().split(".")
    log_file_name = dt.replace(":", "_")
    log_file = log_file_name + ".log"
    raw_log_file = log_file_name + ".raw.log"

    if not log and verbose >= 0:
        print("Logging disabled")
    elif log and not log_all and verbose >= 0:
        print("Logging all disabled; will only give infeasible cases + summary info")

    # keep all generated InputSets
    tests: List[TestCase] = [None]*n_tests
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
            printProgressBar(i_test, n_tests, prefix='Progress:',
                             suffix='Complete', length=80)
        # workers
        k = random.randint(k_lim // 2, k_lim)
        s_max = [random.randint(0, s_max_lim) for _ in range(k)]
        s_max_avg = mean(s_max)
        c = [round(random.random() * c_lim, 2) for _ in range(k)]
        # data set
        n = random.randint(
            max(1, floor(k * s_max_avg * n_factor)), ceil(k * s_max_avg))
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
                workers, data_set, beta, s_min)
            assigned = [w for w in workers if w.id in assigned_ids]

            stats = Stats(input_set, workers)

            if verbose >= 1:
                print(stats)
            elif verbose >= 2:
                print("\nAll workers:")
                print(workers)
                print("\nAssigned workers:")
                print(assigned)
        except (InsufficientWorkersError, InsufficientCapacityError, InsufficientDataError, InfeasibleWorkerCapacityError, AssignmentError, ValueError) as e:
            infeasible[e.__class__.__name__].append(i_test)
            error = repr(e)
            if verbose >= 1:
                print(e)
        finally:
            feasible = error is None
            test_case = TestCase(input_set, feasible,
                                 workers, stats, timing, error)
            tests[i_test] = test_case

    feasible_tests = [x for x in enumerate(tests) if x[1].feasible]
    n_feasible = len(feasible_tests)
    feasible_pct = round(100 * n_feasible / max(n_tests, 1))
    info = [
        "test_random output log",
        "n_tests={}, seed={}, verbose={}, log={}, log_file={}".format(
            n_tests, seed, verbose, log, log_file),
        "Feasible distributions found: {} ({}%)".format(
            n_feasible, feasible_pct),
        "Infeasible distributions encountered:",
        "".join(map(lambda x: "{}: {}, ".format(
            x[0], len(x[1])), infeasible.items())),
        "",
    ]
    if verbose >= 0:
        print("\nTests complete")
        print("\n".join(list(info)))

    if (log):
        if verbose >= 0:
            print("Writing logs...")
        with open(log_file, "w") as f:
            templ = "  TEST {}: {{\n{}\n  }}"
            if log_all:
                # add feasible tests
                info.append("Feasible Tests ({}):".format(n_feasible))
                info += [templ.format(x[0], x[1].__str__(4))
                         for x in feasible_tests]
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
                raw_log = [templ.format(i, tests[i].__str__(2))
                           for i in range(n_tests)]
                f.writelines("\n".join(list(raw_log)))

                if verbose >= 0:
                    print("Wrote raw log dump to '{}'".format(raw_log_file))

    return tests


if __name__ == "__main__":
    # System parameters
    # MARK: Workers
    # number of available workers in the system
    # The below is a case from test_random(10000, seed=100) (i think that's the right seed, may be wrong)
    # Worker 8 and 10 are given s_max so as to not be selected
    k = 2
    # upper bound estimates of workers (must be of length k)
    # we are assuming these are already provided from Duncan's piece based on benchmarks + total training time param
    s_max = [79, 71]
    # worker costs (must be of length k)
    c = [15.78, 9.38]

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
    n = 1
    # data set is temporarily integers
    data_set = [i for i in range(n)]

    # MARK: algorithm parameters
    # minimum number of employed workers
    beta = 1
    # minimum assignable amount of work
    s_min = 0

    input_set = InputSet(n, beta, s_min, k, s_max, c)

    [assigned_ids, timing] = assign_work(workers, data_set, beta, s_min)
    assigned = [w for w in workers if w.id in assigned_ids]

    stats = Stats(input_set, workers)

    workers.sort(key=lambda w: w.c)

    print("*** RESULTS ***")
    print(stats)
    print("\nAll workers:")
    print(workers)
    print("\nAssigned workers:")
    print(assigned)
    print("\nTiming info:")
    print(timing)
