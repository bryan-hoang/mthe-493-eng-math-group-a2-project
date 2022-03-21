from typing import List
from itertools import combinations

from model import Worker
from error import InfeasibleWorkerCapacityError


# Sets whether we use subset brute-force method or optimized method (WIP)
USE_SUBSET = True


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
    pass
    # eligible = [w for w in workers if w.s_max > 0 and w.s_max >= s_min]
    # return eligible
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
    # max_employable = n // s_min
    # # it's important that these lists remain sorted by c
    # selected = eligible[:max_employable]
    # unselected = eligible[max_employable:]
    # # check if we've already got the best selection
    # if max_employable > len(selected) or get_capacity(selected) >= n:
    #     return selected
    # # otherwise, we need to do some swapping, starting with the most expensive selected workers
    # # this doesn't guarantee an optimal solution; we will (on each pass) swap the most expensive

    # # WE SHOULD EXAMINE: how much capacity do we gain vs. how much does it cost?
    # # compare each selected worker to each unselected worker. calculate, for each pair, the capacity gained, and the added expense.
    # # then, check if there is any single swap that will meet required capacity
    # #  -> if one such swap exists, filter all swaps by those that will meet requirement. Then find cheapest one and perform it
    # #  -> otherwise, we need to determine which swaps to consider, and follow multiple paths. Could:
    # #     -> simulate one swap for each selected? (breadth first)
    # #     -> simulate taking swap which increases capacity by the most, or swap which increases cost by the least?
    # for swap_idx in range(len(selected) - 1, -1, -1):
    #     did_swap = False
    #     # pop swappable worker
    #     to_unselect = selected.pop(swap_idx)
    #     # capacity of remaining selected workers
    #     partial_capacity = get_capacity(selected)
    #     required_s_max = n - partial_capacity
    #     # find an unselected worker with required s_max
    #     for w in unselected:
    #         if w.s_max >= required_s_max:
    #             insert_to_sorted_list(selected, w, key=lambda x: x.c)
    #             did_swap = True
    #             break
    #     # if we didn't swap, add to_unselect back to selected
    #     if not did_swap:
    #         selected.insert(swap_idx, to_unselect)


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
