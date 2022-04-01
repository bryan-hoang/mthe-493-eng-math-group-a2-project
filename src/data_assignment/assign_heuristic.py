import time
from typing import List, Tuple

from .error import (
    AssignmentError,
    InsufficientCapacityError,
    InsufficientDataError,
    InsufficientWorkersError,
)
from .model import InputSet, Timing, Worker
from .stats import Stats
from .utils import get_capacity, get_employable_workers

# Should the timing in assign_work be enabled?
ENABLE_NATIVE_TIMING = False


def assign_work_heuristic(
    workers: List[Worker], data_set: List, beta: int, s_min: int
) -> Tuple[List[int], Timing]:
    """
    Assigns a data set to a list of workers by partitioning based on properties
    of workers (s_max, c) and parameters of the algorithm (beta, s_min)

    Returns tuple:
      [0]: list of worker IDs that received non-zero work
      [1]: timing/duration info
    """
    # TIMING start
    duration = Timing(0, 0, 0, 0, 0, 0)
    if ENABLE_NATIVE_TIMING:
        timing = [round(time.time_ns() / 1000)]

    # derive some properties from arguments
    n = len(data_set)
    k = len(workers)

    # CHECK 0: n > 0
    if n <= 0:
        raise ValueError("Must have at least one data element (n: {})".format(n))
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
        items = data_set[i_data : i_data + x]
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
                "Unknown error occurred while attempting reassignment (ran out of workers that can donate)"
            )
        source = employable[donor_worker_idx]
        # grab some reassignable data from the last complete worker
        n_available = source.num_assigned - s_min
        # keep reassigning from the source until it has none left to donate, and we still need to donate
        while n_available > 0 and num_complete_workers_assigned < max(
            num_workers_assigned, beta
        ):
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
            "Invalid s_max list: expected length: {}; actual length: {}".format(
                k, len(s_max)
            )
        )
    elif len(c) != k:
        raise ValueError(
            "Invalid c list: expected length: {}; actual length: {}".format(k, len(c))
        )

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

    [assigned_ids, timing] = assign_work_heuristic(workers, data_set, beta, s_min)
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
