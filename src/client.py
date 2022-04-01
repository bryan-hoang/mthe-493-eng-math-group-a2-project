import asyncio
import random
import time
from math import floor
from typing import List
import argparse

import torch
from axon import client, config, discovery
from torchvision import transforms
from torchvision.datasets import MNIST

import numpy as np

import common
from common import TwoNN, get_accuracy, set_parameters
from data_assignment.assign_heuristic import assign_work_heuristic
from data_assignment.error import (
    AssignmentError,
    InfeasibleWorkerCapacityError,
    InsufficientCapacityError,
    InsufficientDataError,
    InsufficientWorkersError,
)
from data_assignment.model import Worker
from environment import (
    get_env_batch_size,
    get_env_beta,
    get_env_default_fee,
    get_env_device,
    get_env_fee_type,
    get_env_fees,
    get_env_max_time,
    get_env_num_benchmark,
    get_env_num_global_cycles,
    get_env_s_min,
    get_env_weight_type,
)
from log import dump_logs

nb_ip = None

device = get_env_device()

# importing data
test_transform = train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST("./data", train=True, download=True, transform=train_transform)
test_set = MNIST("./data", train=False, download=True, transform=test_transform)

x_train_raw, y_train_raw = map(list, zip(*train_set))
x_test_raw, y_test_raw = map(list, zip(*test_set))

# formatting data
x_train = torch.cat(x_train_raw).reshape([-1, 784])
x_test = torch.cat(x_test_raw).reshape([-1, 784])

y_train = torch.tensor(y_train_raw, dtype=torch.long)
y_test = torch.tensor(y_test_raw, dtype=torch.long)

# defines the central model, as well as the criterion
net = TwoNN().to(device)
criterion = torch.nn.CrossEntropyLoss()

# this function aggregates parameters from workers
def aggregate_parameters(param_list, weights):

    num_clients = len(param_list)
    avg_params = []

    for i, params in enumerate(param_list):

        if i == 0:
            for p in params:
                avg_params.append(p.clone() * weights[i])

        else:
            for j, p in enumerate(params):
                avg_params[j].data += p.data * weights[i]

    return avg_params


# gets the accuracy and loss of a neural net on testing data
def val_evaluation(net, x_test, y_test, params):
    BATCH_SIZE = params.get("BATCH_SIZE", get_env_batch_size())

    num_test_batches = x_test.shape[0] // BATCH_SIZE

    loss = 0
    acc = 0

    net = net.to(device)
    for batch_number in range(num_test_batches):
        x_batch = x_test[
            BATCH_SIZE * batch_number : BATCH_SIZE * (batch_number + 1)
        ].to(device)
        y_batch = y_test[
            BATCH_SIZE * batch_number : BATCH_SIZE * (batch_number + 1)
        ].to(device)
        y_hat = net.forward(x_batch)

        loss += criterion(y_hat, y_batch).item()
        acc += get_accuracy(y_hat, y_batch).item()

    # normalizing the loss and accuracy
    loss = loss / num_test_batches
    acc = acc / num_test_batches

    return loss, acc


def get_fees(n, params):
    FEE_TYPE = params.get("FEE_TYPE", get_env_fee_type())
    DEFAULT_FEE = params.get("DEFAULT_FEE", get_env_default_fee())

    print("Using {} fee type".format(FEE_TYPE))

    if FEE_TYPE == "random":
        return [random.randint(1, 20) for _ in range(n)]
    elif FEE_TYPE == "linear":
        return [i + 1 for i in range(n)]
    elif FEE_TYPE == "specific":
        return params.get("FEES", get_env_fees(n))
    else:
        # default to constant
        return [DEFAULT_FEE] * n


def get_weight_type(params):
    WEIGHT_TYPE = params.get("WEIGHT_TYPE", get_env_weight_type())

    print("Using {} weight type".format(WEIGHT_TYPE))

    if WEIGHT_TYPE == "xavier":
        return common.init_weights_xavier
    elif WEIGHT_TYPE == "kaiming":
        return common.init_weights_kaiming
    else:
        # default to orthogonal
        return common.init_weights_orthogonal


async def main(arg_nb_ip=None, params=None):
    global nb_ip

    if params is None:
        params = {}

    # Set nb_ip from args (if provided) or broadcast IP otherwise
    if arg_nb_ip:
        nb_ip = arg_nb_ip
    else:
        # grabs notice board ip for discovery use
        axon_local_ips = await discovery.broadcast_discovery(
            num_hosts=1, port=config.comms_config.notice_board_port
        )

        nb_ip = axon_local_ips.pop()

    log_dict = {}

    # ML-related params
    BATCH_SIZE = params.get("BATCH_SIZE", get_env_batch_size())
    NUM_GLOBAL_CYCLES = params.get("NUM_GLOBAL_CYCLES", get_env_num_global_cycles())
    NUM_BENCHMARK = params.get("NUM_BENCHMARK", get_env_num_benchmark())
    # Data-assignment-related params
    BETA = params.get("BETA", get_env_beta())
    S_MIN = params.get("S_MIN", get_env_s_min())
    MAX_TIME = params.get("MAX_TIME", get_env_max_time())
    max_time_per_cycle = MAX_TIME / NUM_GLOBAL_CYCLES
    # Fees
    FEE_TYPE = params.get("FEE_TYPE", get_env_fee_type())
    WEIGHT_TYPE = params.get("WEIGHT_TYPE", get_env_weight_type())

    print("*** System Parameters ***")
    print(
        "BATCH_SIZE={} NUM_GLOBAL_CYCLES={} NUM_BENCHMARK={}".format(
            BATCH_SIZE, NUM_GLOBAL_CYCLES, NUM_BENCHMARK
        )
    )
    print(
        "BETA={} S_MIN={} MAX_TIME={} FEE_TYPE={}".format(
            BETA, S_MIN, MAX_TIME, FEE_TYPE
        )
    )

    log_dict["BATCH_SIZE"] = BATCH_SIZE
    log_dict["NUM_GLOBAL_CYCLES"] = NUM_GLOBAL_CYCLES
    log_dict["NUM_BENCHMARK"] = NUM_BENCHMARK
    log_dict["BETA"] = BETA
    log_dict["S_MIN"] = S_MIN
    log_dict["MAX_TIME"] = MAX_TIME
    log_dict["max_time_per_cycle"] = max_time_per_cycle
    log_dict["FEE_TYPE"] = FEE_TYPE
    log_dict["WEIGHT_TYPE"] = WEIGHT_TYPE

    print("\n*** Reinitialize net ***")
    reinit_weights_fcn = get_weight_type(params)
    net.apply(reinit_weights_fcn)

    # find and connect to workers
    worker_ips = discovery.get_ips(ip=nb_ip)

    # instantiates remote worker objects, with which we can call rpcs on each worker
    axon_workers = [client.RemoteWorker(ip) for ip in worker_ips]

    print("\n*** Benchmarking Workers ***")
    # start benchmarks in each worker
    benchmark_coros = []
    for w in axon_workers:
        benchmark_coros.append(w.rpcs.benchmark(NUM_BENCHMARK))

    # wait for each worker to finish their benchmark
    benchmark_scores = await asyncio.gather(*benchmark_coros)
    log_dict["benchmark_scores"] = {
        x[0]: x[1] for x in zip(worker_ips, benchmark_scores)
    }
    # calculates the number of batches each worker should be assigned
    s_max_batches = [floor(max_time_per_cycle * b) for b in benchmark_scores]

    # calc total number of batches
    total_batches = x_train.shape[0] // BATCH_SIZE
    log_dict["total_batches"] = total_batches

    fees = get_fees(len(axon_workers), params)

    workers: List[Worker] = []
    for i, w in enumerate(axon_workers):
        ip = worker_ips[i]
        s_max = s_max_batches[i]
        wage = fees[i]
        new_worker = Worker(s_max, wage, ip, w)
        workers.append(new_worker)

    log_dict["workers"] = workers

    print("\n*** Setting Worker Fees ***")

    # set wages in each worker
    minwage_coros = []
    for w in workers:
        print("Worker {} fee: {}".format(w.id, w.c))
        minwage_coros.append(w.axon_worker_ref.rpcs.set_minimum_wage(w.c))

    await asyncio.gather(*minwage_coros)

    print("\n*** Sending Data to Workers ***")
    send_data_start = time.time()

    # The Worker model assumes we partition the set of batches
    dataset = [x for x in range(total_batches)]
    allocations_pending = []
    try:
        [employed_workers, assignment_timing_stats] = assign_work_heuristic(
            workers, dataset, BETA, S_MIN
        )
        print(
            "Assigning data to {} / {} workers".format(
                len(employed_workers), len(workers)
            )
        )
        for w in workers:
            if w.id in employed_workers:
                # Extract the dataset here
                # idxs = torch.randperm(x_train.shape[0])[0 : BATCH_SIZE * w.num_assigned]
                # w.assigned_work will be a List of indexes of batches, samples_indices is a 2D list
                batch_samples = list(
                    map(
                        lambda batch_idx: list(
                            range(batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE)
                        ),
                        w.assigned_work,
                    )
                )
                sample_indices = np.array(batch_samples).flatten()
                num_samples = sample_indices.shape[0]
                x_data = x_train[sample_indices]
                y_data = y_train[sample_indices]

                print(
                    "Assigning worker {} {} batches ({} samples)".format(
                        w.id, num_samples // BATCH_SIZE, num_samples
                    )
                )

                allocations_pending.append(
                    w.axon_worker_ref.rpcs.set_training_data(x_data, y_data)
                )
    except (
        InsufficientWorkersError,
        InsufficientCapacityError,
        InsufficientDataError,
        InfeasibleWorkerCapacityError,
        AssignmentError,
        ValueError,
    ) as e:
        print("Infeasible")
        raise (e)

    await asyncio.gather(*allocations_pending)

    send_data_duration = time.time() - send_data_start
    log_dict["send_data_duration"] = send_data_duration

    # Calculate how many batches were actually assigned
    total_batches_assigned = sum(map(lambda w: w.num_assigned, workers))
    log_dict["total_batches_assigned"] = total_batches_assigned
    # determine which workers were actually assigned data
    assigned_workers = list(filter(lambda w: w.num_assigned > 0, workers))

    print("\n*** Training ***")

    model_stats = {}
    log_dict["model_stats"] = model_stats
    # evaluate parameters
    loss, acc = val_evaluation(net, x_test, y_test, params)
    print("Network loss, accuracy prior to training: {}, {}".format(loss, acc))
    model_stats["0"] = {"loss": loss, "acc": acc}
    log_dict["model_pre_train_loss"] = loss
    log_dict["model_pre_train_acc"] = acc

    training_start = time.time()

    for i in range(NUM_GLOBAL_CYCLES):
        # some workers don't have a GPU and the device that a tensor is on will be serialized, so we've gotta move the network to CPU before transmitting parameters to worker
        net.to("cpu")

        # local updates
        local_update_coros = []
        for w in assigned_workers:
            local_update_coros.append(
                w.axon_worker_ref.rpcs.local_update(list(net.parameters()))
            )

        # waits for local updates to complete
        worker_params = await asyncio.gather(*local_update_coros)

        net.to(device)

        # aggregates parameters
        weights = [w.num_assigned / total_batches_assigned for w in workers]
        new_params = aggregate_parameters(worker_params, weights)

        # sets the central model to the new parameters
        set_parameters(net, new_params)

        # evaluate new parameters
        loss, acc = val_evaluation(net, x_test, y_test, params)
        print(
            "({}/{}) Network loss, accuracy: {}, {}".format(
                i + 1, NUM_GLOBAL_CYCLES, loss, acc
            )
        )
        model_stats[str(i + 1)] = {"loss": loss, "acc": acc}

    log_dict["model_post_train_loss"] = loss
    log_dict["model_post_train_acc"] = acc

    training_duration = time.time() - training_start
    log_dict["training_duration"] = training_duration

    timing_logs_coros = []
    for w in assigned_workers:
        timing_logs_coros.append(w.axon_worker_ref.rpcs.return_and_clear_timing_logs())

    # wait for timing logs to be returned from each worker
    timing_logs = await asyncio.gather(*timing_logs_coros)
    timing_logs = {x[0].id: x[1] for x in zip(assigned_workers, timing_logs)}
    log_dict["timing_logs"] = timing_logs

    print("\n*** Results ***")

    # Worker model computes total cost as w.cost = ceil(w.num_assigned / w.batch_size) * w.c
    total_cost_per_cycle = sum(map(lambda w: w.cost, workers))
    log_dict["total_cost_per_cycle"] = total_cost_per_cycle
    total_cost = total_cost_per_cycle * NUM_GLOBAL_CYCLES
    log_dict["total_cost"] = total_cost

    for w in workers:
        template = "Worker {} assigned {} batches (= {} samples); fee per cycle: {}; total fee: {}"
        msg = template.format(
            w.id,
            w.num_assigned,
            w.num_assigned * BATCH_SIZE,
            w.cost,
            w.cost * NUM_GLOBAL_CYCLES,
        )
        print(msg)
    print("Total fee per cycle:", total_cost_per_cycle)
    print("Total fee ({} cycles): {}".format(NUM_GLOBAL_CYCLES, total_cost))
    print("Data assignment duration:", send_data_duration)
    print("Training duration:", training_duration)

    # Dump log_dict to CSVs
    print("\n*** Writing Logs ***")
    try:
        dump_logs(log_dict, params)
        print("Success!")
    except Exception as e:
        print("Failed!")
        print(e)

    return log_dict


# Instantiate the parser
parser = argparse.ArgumentParser(description="Optional app description")
parser.add_argument("--nb-ip", type=str, help="Notice board host IP")

if __name__ == "__main__":
    args = parser.parse_args()
    asyncio.run(main(arg_nb_ip=args.nb_ip))
