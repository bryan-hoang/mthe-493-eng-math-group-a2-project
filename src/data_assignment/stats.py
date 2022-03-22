from statistics import StatisticsError, mean, variance
from typing import List, Union

from .model import InputSet, Worker
from .utils import get_employable_workers


class Stats:

    stat_list = [
        "n",
        "beta",
        "s_min",
        "k",
        "total_assigned",
        "total_cost",
        "k_employable",
        "max_employable",
        "employable_total_ratio",
        "k_assigned",
        "assigned_total_ratio",
        "assigned_employed_ratio",
        "avg_utilization",
        "avg_utilization_employable",
        "avg_utilization_assigned",
        "avg_c",
        "avg_c_employable",
        "avg_c_assigned",
        "var_c",
        "var_c_employable",
        "var_c_assigned",
        "avg_s_max",
        "avg_s_max_employable",
        "avg_s_max_assigned",
        "var_s_max",
        "var_s_max_employable",
        "var_s_max_assigned",
        "avg_cost",
        "avg_cost_employable",
        "avg_cost_assigned",
        "var_cost",
        "var_cost_employable",
        "var_cost_assigned",
        "avg_assigned",
        "avg_assigned_employable",
        "avg_assigned_assigned",
        "var_assigned",
        "var_assigned_employable",
        "var_assigned_assigned",
    ]
    stats_pct_list = [
        "employable_total_ratio",
        "assigned_total_ratio",
        "assigned_employed_ratio",
        "avg_utilization",
        "avg_utilization_employable",
        "avg_utilization_assigned",
    ]

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

    def __init__(
        self, input_set: InputSet, workers: List[Worker], employable=None
    ):
        self.input_set = input_set
        self.workers = workers
        self.employable = employable or get_employable_workers(
            workers, input_set.s_min, input_set.n
        )
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
        self.max_employable = (
            self.n // self.s_min if self.s_min > 0 else self.k_employable
        )
        self.employable_total_ratio = self.k_employable / self.k

        self.k_assigned = len(self.assigned)
        self.assigned_total_ratio = self.k_assigned / self.k
        self.assigned_employed_ratio = self.k_assigned / self.k_employable

        self.avg_utilization = Stats.get_avg_utilization(self.workers)
        self.avg_utilization_employable = Stats.get_avg_utilization(
            self.employable
        )
        self.avg_utilization_assigned = Stats.get_avg_utilization(
            self.assigned
        )

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
