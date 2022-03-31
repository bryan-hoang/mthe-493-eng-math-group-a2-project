import os
import pathlib
from environment import (
    get_env_batch_size,
    get_env_log_id,
    get_env_logs,
    get_env_num_global_cycles,
)

SPECIAL_LOGS = [
    "workers",
    "timing_logs",
    "model_stats",
    "benchmark_scores",
]


def make_workers_csv(workers, benchmark_scores, params):
    BATCH_SIZE = params.get("BATCH_SIZE", get_env_batch_size())
    NUM_GLOBAL_CYCLES = params.get("NUM_GLOBAL_CYCLES", get_env_num_global_cycles())

    headers = "ip,benchmark_batches_per_sec,s_max,cost_per_batch,cost_per_cycle,total_cost,batches_assigned,samples_assigned"
    values = []
    for w in workers:
        benchmark_score = benchmark_scores.get(w.id, -1)
        val = "{},{},{},{},{},{},{},{}".format(
            w.id,
            benchmark_score,
            w.s_max,
            w.c,
            w.cost,
            w.cost * NUM_GLOBAL_CYCLES,
            w.num_assigned,
            w.num_assigned * BATCH_SIZE,
        )
        values.append(val)

    return "\n".join([headers] + values)


def make_model_stats_csv(model_stats):
    headers = "loss,accuracy"
    values = []
    for i in model_stats:
        val = "{},{}".format(model_stats[i]["loss"], model_stats[i]["acc"])
        values.append(val)

    return "\n".join([headers] + values)


def make_timing_logs_csv(timing_logs, params):
    NUM_GLOBAL_CYCLES = params.get("NUM_GLOBAL_CYCLES", get_env_num_global_cycles())

    headers = ""
    values = ["" for _ in range(NUM_GLOBAL_CYCLES)]

    for id in timing_logs:
        headers += "{},".format(id)
        times = timing_logs[id]
        for i in range(NUM_GLOBAL_CYCLES):
            val = times[i] if i < len(times) else "N/A"
            values[i] += "{},".format(val)

    return "\n".join([headers] + values)


def dump_logs(log_dict, params):
    log_base = params.get("LOGS", get_env_logs())
    log_id = params.get("LOG_ID", get_env_log_id())
    pathlib.Path(log_base).mkdir(exist_ok=True, parents=True)

    stats_dir = os.path.join(log_base, "stats")
    workers_dir = os.path.join(log_base, "workers")
    model_dir = os.path.join(log_base, "model")
    timing_dir = os.path.join(log_base, "timing")
    pathlib.Path(stats_dir).mkdir(exist_ok=True, parents=True)
    pathlib.Path(workers_dir).mkdir(exist_ok=True, parents=True)
    pathlib.Path(model_dir).mkdir(exist_ok=True, parents=True)
    pathlib.Path(timing_dir).mkdir(exist_ok=True, parents=True)

    csv_filename = "{}.csv".format(log_id)

    stats_path = os.path.join(stats_dir, csv_filename)
    workers_path = os.path.join(workers_dir, csv_filename)
    model_path = os.path.join(model_dir, csv_filename)
    timing_path = os.path.join(timing_dir, csv_filename)

    workers = log_dict["workers"]
    benchmark_scores = log_dict["benchmark_scores"]
    timing_logs = log_dict["timing_logs"]
    model_stats = log_dict["model_stats"]

    header = ""
    values = ""

    for k in log_dict:
        if k not in SPECIAL_LOGS:
            header += "{},".format(k)
            values += "{},".format(log_dict[k])

    with open(stats_path, "w") as f:
        f.write(header + "\n" + values)
    with open(workers_path, "w") as f:
        output = make_workers_csv(workers, benchmark_scores, params)
        f.write(output)
    with open(model_path, "w") as f:
        output = make_model_stats_csv(model_stats)
        f.write(output)
    with open(timing_path, "w") as f:
        output = make_timing_logs_csv(timing_logs, params)
        f.write(output)

