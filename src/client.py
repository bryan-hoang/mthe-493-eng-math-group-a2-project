import asyncio
import random
import time
from math import ceil, floor
from typing import List

import torch
from axon import client, config, discovery
from torchvision import transforms
from torchvision.datasets import MNIST

from common import TwoNN, get_accuracy, set_parameters
from data_assignment.assign import assign_work
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
    get_env_device,
    get_env_max_time,
    get_env_num_benchmark,
    get_env_num_global_cycles,
    get_env_s_min,
)

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
def val_evaluation(net, x_test, y_test):
    BATCH_SIZE = get_env_batch_size()

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


async def main():
    global nb_ip

    # ML-related params
    BATCH_SIZE = get_env_batch_size()
    NUM_GLOBAL_CYCLES = get_env_num_global_cycles()
    NUM_BENCHMARK = get_env_num_benchmark()
    # Data-assignment-related params
    BETA = get_env_beta()
    S_MIN = get_env_s_min()
    MAX_TIME = get_env_max_time()

    # grabs notice board ip for discovery use
    axon_local_ips = await discovery.broadcast_discovery(
        num_hosts=1, port=config.comms_config.notice_board_port
    )

    nb_ip = axon_local_ips.pop()

    # find and connect to workers
    worker_ips = discovery.get_ips(ip=nb_ip)

    # instantiates remote worker objects, with which we can call rpcs on each worker
    axon_workers = [client.RemoteWorker(ip) for ip in worker_ips]

    print("benchmarking workers")
    # start benchmarks in each worker
    benchmark_coros = []
    for w in axon_workers:
        benchmark_coros.append(w.rpcs.benchmark(NUM_BENCHMARK))

    # wait for each worker to finish their benchmark
    benchmark_scores = await asyncio.gather(*benchmark_coros)
    # calculates the number of data batches each worker should be assigned
    # total_batches = (x_train.shape[0]//num_global_cycles)//BATCH_SIZE
    # total_batches = 6000//BATCH_SIZE
    s_max_batches = [floor(MAX_TIME * b) for b in benchmark_scores]

    # calculates the number of data batches each worker should be assigned
    total_batches = 6000 // BATCH_SIZE
    normalizing_factor = total_batches / sum(benchmark_scores)
    data_allocation = [round(normalizing_factor * b) for b in benchmark_scores]

    workers: List[Worker] = []
    for i, w in enumerate(axon_workers):
        ip = worker_ips[i]
        s_max = s_max_batches[i] * BATCH_SIZE
        # get random wage between 1 and 20 (inclusive)
        # TODO: Set this to something deterministic/repeatable
        wage = random.randint(1, 20)
        new_worker = Worker(s_max, wage, ip, w, BATCH_SIZE)
        workers.append(new_worker)

    print("setting worker wages")

    # set wages in each worker
    minwage_coros = []
    for w in workers:
        minwage_coros.append(w.axon_worker_ref.rpcs.set_minimum_wage(w.c))

    await asyncio.gather(*minwage_coros)

    print("sending data to workers")

    # The Worker model assumes we partition the entire dataset amongst workers.
    # Since there's an x/y data set, I'll assume we can treat the indices of data elements as the dataset
    # for assign_work. We can later extract that data before assigning work (set_training_data call)
    # TODO: for Jack: this is where we set the 'global dataset'
    n_data = x_train.shape[0]
    dataset = [x for x in range(n_data)]
    allocations_pending = []
    try:
        [employed_workers, assignment_timing_stats] = assign_work(
            workers, dataset, BETA, S_MIN
        )
        print(
            "Assigning data to {} / {} workers".format(
                len(employed_workers), len(workers)
            )
        )
        for w in workers:
            if w.id in employed_workers:
                # TODO: for Jack: ...and this is where we extract the actual training dataset from the global dataset + assigned_work
                # idxs = torch.randperm(x_train.shape[0])[0:w.num_assigned]
                # x_data = x_train[idxs]
                # y_data = y_train[idxs]
                x_data = x_train[w.assigned_work]
                y_data = y_train[w.assigned_work]

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

    # calc total number of samples assignefd
    # total_batches = sum(map(lambda w: w.num_assigned // BATCH_SIZE, workers))
    total_batches = x_train.shape[0] // BATCH_SIZE
    normalizing_factor = total_batches / sum(benchmark_scores)
    data_allocation = [round(normalizing_factor * b) for b in benchmark_scores]

    # evaluate parameters
    loss, acc = val_evaluation(net, x_test, y_test)
    print("network loss and validation prior to training:", loss, acc)

    start_time = time.time()

    for i in range(NUM_GLOBAL_CYCLES):
        print("training index:", i, "out of", NUM_GLOBAL_CYCLES)

        # some workers don't have a GPU and the device that a tensor is on will be serialized, so we've gotta move the network to CPU before transmitting parameters to worker
        net.to("cpu")

        # local updates
        local_update_coros = []
        for w in workers:
            local_update_coros.append(
                w.axon_worker_ref.rpcs.local_update(list(net.parameters()))
            )

        # waits for local updates to complete
        worker_params = await asyncio.gather(*local_update_coros)

        net.to(device)

        # aggregates parameters
        weights = [d / sum(data_allocation) for d in data_allocation]
        new_params = aggregate_parameters(worker_params, weights)

        # sets the central model to the new parameters
        set_parameters(net, new_params)

        # evaluate new parameters
        loss, acc = val_evaluation(net, x_test, y_test)
        print("network loss and validation:", loss, acc)

    elapsed_time = time.time() - start_time

    timing_logs_coros = []
    for w in workers:
        timing_logs_coros.append(w.axon_worker_ref.rpcs.return_and_clear_timing_logs())

    # wait for timing logs to be returned from each worker
    timing_logs = await asyncio.gather(*timing_logs_coros)

    # Worker model computes total cost as w.cost = ceil(w.num_assigned / w.batch_size) * w.c
    total_cost = sum(map(lambda w: w.cost, workers))

    # TODO: This should maybe be change to reference worker.id instead of the index (async evaluation might resolve in any order?)
    for index, log in enumerate(timing_logs):
        print(
            "worker "
            + str(index)
            + " computed "
            + str(len(log))
            + " batches in "
            + str(sum(log))
            + "s"
        )
    print("total elapsed time for job completion " + str(elapsed_time))
    for w in workers:
        template = "Worker {} assigned {} samples (= {} batches); total worker fee: {}"
        msg = template.format(
            w.id, w.num_assigned, ceil(w.num_assigned / w.batch_size), w.cost
        )
        print(msg)
    print("Total fee: " + str(total_cost))


if __name__ == "__main__":
    asyncio.run(main())
