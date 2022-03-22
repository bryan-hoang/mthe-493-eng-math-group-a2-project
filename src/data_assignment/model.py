from typing import List, NamedTuple, Union

from .error import AssignmentError

# Sets whether a print of Worker will display the data actually allocated
LOG_ASSIGNED_WORK = False
# Should InputSet(c=[...], s_max=[...]) be logged?
LOG_INPUT_LISTS = True


class Worker:
    """
    Worker is just a temporary data structure, used to encapsulate the properties
    of a single worker. May or may not be included in final implementation
    """

    id = 0
    s_max = 0
    c = 0
    axon_worker_ref = None

    num_assigned = 0
    assigned_work = []
    cost = 0

    def __init__(
        self, s_max: int, c: float, id=0, axon_worker_ref=None
    ) -> None:
        self.s_max = s_max
        self.c = c
        self.id = id
        self.axon_worker_ref = axon_worker_ref

    def assign(self, items: List) -> None:
        if len(items) > self.s_max:
            raise AssignmentError(
                "Cannot assign {} items to worker with s_max {}".format(
                    len(items), self.s_max
                )
            )
        self.assigned_work = items
        self.num_assigned = len(items)
        self.cost = self.c * len(items)

    def reassign_from(self, source: "Worker", n: int, s_min: int) -> None:
        """Reassigns n assigned elements from source to self"""
        # check that source can afford to give n
        if n > source.num_assigned - s_min:
            template = "Cannot reassign {} elements from worker {} (s_min: {}, max reassignable: {})"
            msg = template.format(
                n, source, s_min, max(0, source.num_assigned - s_min)
            )
            raise AssignmentError(msg)
        # check that self has capacity for n items
        elif n + self.num_assigned > self.s_max:
            template = "Cannot reassign {} elements to worker {} (s_max: {}, max reassignable: {})"
            msg = template.format(
                n, self, self.s_max, self.s_max - self.num_assigned
            )
            raise AssignmentError(msg)
        # check that we're reassigning non-zero items
        elif not n > 0:
            raise ValueError("Cannot reassign {} items".format(n))
        # we know that source can give n items, and self has room to take n items
        source_items = source.assigned_work
        new_source_items = source_items[: len(source_items) - n]
        new_self_items = (
            self.assigned_work + source_items[len(source_items) - n :]
        )
        source.assign(new_source_items)
        self.assign(new_self_items)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if LOG_ASSIGNED_WORK:
            template = "Worker(id: {}, s_max: {}, c: {}, num_assigned: {}, assigned_work: {}, cost: {})"
            return template.format(
                self.id,
                self.s_max,
                self.c,
                self.num_assigned,
                repr(self.assigned_work),
                round(self.cost, 2),
            )
        else:
            template = (
                "Worker(id: {}, s_max: {}, c: {}, num_assigned: {}, cost: {})"
            )
            return template.format(
                self.id,
                self.s_max,
                self.c,
                self.num_assigned,
                round(self.cost, 2),
            )


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
            str_val += "{}={}{}".format(
                log_keys[i], self[i], ", " if not is_last else ""
            )
        str_val += ")"
        return str_val


class Timing(NamedTuple):
    init: float
    get_employable_workers: float
    get_capacity: float
    assign: float
    reassign: float
    total: float
