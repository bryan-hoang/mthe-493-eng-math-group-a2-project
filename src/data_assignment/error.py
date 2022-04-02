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
        msg = template.format(available, max(1, s_min) * beta, beta, s_min)
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
        template = (
            "No subset of {} workers can compute data (data set size: {}, s_min: {})"
        )
        msg = template.format(max_employable, n, s_min)
        super().__init__(msg)


class AssignmentError(Exception):
    __name__ = "AssignmentError"

    def __init__(self, msg) -> None:
        super().__init__(msg)


class GurobiInfeasibleError(Exception):
    __name__ = "GurobiInfeasibleError"

    def __init__(self) -> None:
        msg = "Gurobi model is infeasible"
        super().__init__(msg)


class PuLPInfeasibleError(Exception):
    __name__ = "PuLPInfeasibleError"

    def __init__(self) -> None:
        msg = "PuLP model is infeasible"
        super().__init__(msg)

