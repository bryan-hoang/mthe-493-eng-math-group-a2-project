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

    # weights_type_values = ["xavier", "kaiming", "orthogonal"]

    # s_min_values = [1, 10, 100, 250, 500]
    s_min_values = [1] + list(2 ** x for x in range(5, 9))

    # beta_values = [1, 3, 5]
    # beta_values = [1, 3, 5]
    beta_values = [5]

    num_global_cycles_values = [80]
    # num_global_cycles_values = [10]

    # max_time_values = range(10)
    max_time_values = [600]

    fee_type = "constant"

    weight_type = "xavier"

    param_set = list(
        product(
            # weights_type_values,
            s_min_values,
            beta_values,
            num_global_cycles_values,
            max_time_values,
        )
    )

    prefix_len = len(str(len(param_set)))

    for i, param_tuple in enumerate(param_set):
        print("({}/{}) Setup".format(i, len(param_set)))

        s_min, beta, num_global_cycles, max_time = param_tuple
        log_prefix = str(i).zfill(prefix_len)
        params = {
            "S_MIN": s_min,
            "BETA": beta,
            "NUM_GLOBAL_CYCLES": num_global_cycles,
            "WEIGHT_TYPE": weight_type,
            "MAX_TIME": max_time,
            "FEE_TYPE": fee_type,
            "LOG_ID": log_prefix,
        }

        print("({}/{}) Run".format(i, len(param_set)))

        try:
            log_dict = await client.main(arg_nb_ip, params)
            all_logs[i] = log_dict
        except Exception as e:
            print("error @ run {}".format(i))
            print(e)
            all_logs[i] = str(e)

    print("\n\n*** Done, dumping logs ***")
    try:
        with open("logs/dump.json", "w") as f:
            json.dump(all_logs, f)
        print("Log dump success!")
    except Exception as e:
        print("Error: could not dump logs")
        print(e)


parser = argparse.ArgumentParser(description="Optional app description")
parser.add_argument("--nb-ip", type=str, help="Notice board host IP")

if __name__ == "__main__":
    args = parser.parse_args()
    asyncio.run(run_tests(arg_nb_ip=args.nb_ip))
