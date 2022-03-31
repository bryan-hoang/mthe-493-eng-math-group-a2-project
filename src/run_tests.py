import json
import client
from itertools import product
import argparse
import asyncio

"""
Values in params dict:

BATCH_SIZE
NUM_GLOBAL_CYCLES
NUM_BENCHMARK
BETA
S_MIN
MAX_TIME
FEE_TYPE
DEFAULT_FEE
FEES
LOGS
LOG_ID
"""


async def run_tests(arg_nb_ip=None):
    all_logs = {}

    # s_min_min = 50
    # s_min_max = 500
    # s_min_inc = 50
    # s_min_values = [5, 50, 100, 200, 500]
    s_min_values = [5]

    # beta_min = 1
    # beta_max = 5
    # beta_inc = 2
    # beta_values = [1, 3, 5]
    # beta_values = [1, 3]
    beta_values = [1]

    # num_global_cycles_min = 6
    # num_global_cycles_max = 20
    # num_global_cycles_inc = 2
    # num_global_cycles_values = [1, 5, 10, 15, 20]
    num_global_cycles_values = [10]

    # max_time_min = 10
    # max_time_max = 100
    # max_time_inc = 10
    # max_time_values = [5, 10, 15, 20, 30, 60]
    max_time_values = [60]

    fee_type = "constant"

    param_set = list(
        product(s_min_values, beta_values, num_global_cycles_values, max_time_values)
    )

    prefix_len = len(str(len(param_set)))

    for i, param_tuple in enumerate(param_set):
        print("({}) Setup".format(i))

        s_min, beta, num_global_cycles, max_time = param_tuple
        log_prefix = str(i).zfill(prefix_len)
        params = {
            "S_MIN": s_min,
            "BETA": beta,
            "NUM_GLOBAL_CYCLES": num_global_cycles,
            "MAX_TIME": max_time,
            "FEE_TYPE": fee_type,
            "LOG_ID": log_prefix,
        }

        print("({}) Run".format(i))

        try:
            log_dict = await client.main(arg_nb_ip, params)
            all_logs[i] = log_dict
        except Exception as e:
            print("error @ run {}".format(i))
            print(e)
            all_logs[i] = str(e)

    print("Done, dumping logs")
    with open("dump.json", "w") as f:
        json.dump(all_logs, f)


parser = argparse.ArgumentParser(description="Optional app description")
parser.add_argument("--nb-ip", type=str, help="Notice board host IP")

if __name__ == "__main__":
    args = parser.parse_args()
    asyncio.run(run_tests(arg_nb_ip=args.nb_ip))
