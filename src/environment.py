import os
import time

from dotenv import load_dotenv
import torch

# Load environment variables from `.env`.
load_dotenv()


def get_env_beta():
    """Returns BETA, the minimum number of workers that must be assigned non-zero work."""
    return int(os.environ.get("BETA", 1))


def get_env_s_min():
    """Returns S_MIN, the minimum quantity of work that must be assigned to a worker, if it is receiving non-zero slices."""
    return int(os.environ.get("S_MIN", 10))


def get_env_max_time():
    """Returns MAX_TIME, the duration we want workers to compute for (seconds)"""
    return float(os.environ.get("MAX_TIME", 30))


def get_env_fee_type():
    """
    Returns FEE_TYPE

    'random': all workers have random fees between 1 and 20
    'constant': all workers have fees of 1
    'linear': fees are 1, 2, 3, 4, ...
    'specific': fees are specified by FEES, a comma-seperated list of floats
    """
    return os.environ.get("FEE_TYPE", "constant")


def get_env_fees(n=None):
    """
    Returns FEES, padded to length n

    FEES should be comma-seperated list of floats
    """
    fees_str = os.environ.get("FEES", "")
    fees = []
    for fee_str in fees_str.split(","):
        try:
            fee = float(fee_str)
            fees.append(fee)
        except ValueError:
            fees.append(get_env_default_fee())

    # pad to n
    fees += [get_env_default_fee()] * (n - len(fees))

    return fees


def get_env_default_fee():
    try:
        return float(os.environ.get("DEFAULT_FEE", 1))
    except ValueError:
        return 1.0


def get_env_num_benchmark():
    """Returns NUM_BENCHMARK, number of fake batches for workers to compute in benchmark."""
    return int(os.environ.get("NUM_BENCHMARK", 1000))


def get_env_num_global_cycles():
    """Returns NUM_GLOBAL_CYCLES, the number of times each worker should perform their local update"""
    return int(os.environ.get("NUM_GLOBAL_CYCLES", 10))


def get_env_batch_size():
    """Returns BATCH_SIZE, the number of samples in each batch"""
    return int(os.environ.get("BATCH_SIZE", 32))


def get_env_allow_gpu_device():
    """Returns ALLOW_GPU_DEVICE, indicating whether CUDA-enabled GPUs can be used for training"""
    return bool(os.environ.get("ALLOW_GPU_DEVICE", True))


def get_env_device():
    if get_env_allow_gpu_device():
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        return "cpu"


def get_env_logs():
    """Returns LOGS, path where logs should be dumped"""
    return os.environ.get("LOGS", "logs")


def get_env_log_id():
    return os.environ.get("LOG_ID", str(time.time_ns() // 1000))
