import os

from dotenv import load_dotenv
import torch

# Load environment variables from `.env`.
load_dotenv()


def get_env_beta():
    """Returns BETA, the minimum number of workers that must be assigned non-zero work."""
    return int(os.environ.get("BETA", 1))


def get_env_s_min():
    """Returns S_MIN, the minimum quantity of work that must be assigned to a worker, if it is receiving non-zero slices."""
    return int(os.environ.get("S_MIN", 1))


def get_env_max_time():
    """Returns MAX_TIME, the duration we want workers to compute for (seconds)"""
    return int(os.environ.get("MAX_TIME", 800))


def get_env_num_benchmark():
    """Returns NUM_BENCHMARK, number of fake batches for workers to compute in benchmark."""
    return int(os.environ.get("NUM_BENCHMARK", 500))


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
