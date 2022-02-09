from axon import config, discovery, client
from common import TwoNN, set_parameters, get_accuracy
# FIXME(bryan-hoang): Remove tensorflow dependency for dataset.
from keras.datasets import mnist
import time
import asyncio, torch

num_global_cycles = 10
nb_ip = None
BATCH_SIZE = 32

device = 'cpu'
if torch.cuda.is_available(): device = 'cuda:0'

# importing data
# FIXME(bryan-hoang): Remove tensorflow dependency for dataset.
raw_data = mnist.load_data()

x_train_raw = raw_data[0][0]
y_train_raw = raw_data[0][1]
x_test_raw = raw_data[1][0]
y_test_raw = raw_data[1][1]

# formatting data
x_train = torch.tensor(x_train_raw, dtype=torch.float32).reshape([-1, 784])
x_test = torch.tensor(x_test_raw, dtype=torch.float32).reshape([-1, 784])

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

		if (i == 0):
			for p in params:
				avg_params.append(p.clone()*weights[i])

		else:
			for j, p in enumerate(params):
				avg_params[j].data += p.data*weights[i]

	return avg_params

# gets the accuracy and loss of a neural net on testing data
def val_evaluation(net, x_test, y_test):

	num_test_batches = x_test.shape[0]//BATCH_SIZE

	loss = 0
	acc = 0

	net = net.to(device)

	for batch_number in range(num_test_batches):
		x_batch = x_test[BATCH_SIZE*batch_number : BATCH_SIZE*(batch_number+1)].to(device)
		y_batch = y_test[BATCH_SIZE*batch_number : BATCH_SIZE*(batch_number+1)].to(device)

		y_hat = net.forward(x_batch)

		loss += criterion(y_hat, y_batch).item()
		acc += get_accuracy(y_hat, y_batch).item()

	# normalizing the loss and accuracy
	loss = loss/num_test_batches
	acc = acc/num_test_batches

	return loss, acc

async def main():
	global nb_ip

	# grabs notice board ip for discovery use
	axon_local_ips = await discovery.broadcast_discovery(num_hosts=1, port=config.comms_config.notice_board_port)

	nb_ip = axon_local_ips.pop()

	# starts the RVL
	await client.start_client()

	# find and connect to workers
	worker_ips = discovery.get_ips(ip=nb_ip)

	# instantiates remote worker objects, with which we can call rpcs on each worker
	workers = [client.RemoteWorker(ip) for ip in worker_ips]

	print('setting worker wages')

	# Hardcode experiments for wages and max price, both normalized per batch
	job_max_price_per_batch = 10
	wages_per_batch = [5]
	paid_out_for_job = [0]

	# set wages in each worker
	minwage_coros = []
	for index, w in enumerate(workers):
		minwage_coros.append(w.rpcs.set_minimum_wage(wages_per_batch[index]))

	await asyncio.gather(*minwage_coros)

	print('benchmarking workers')

	# start benchmarks in each worker
	benchmark_coros = []
	for w in workers:
		benchmark_coros.append(w.rpcs.benchmark(1000))

	# wait for each worker to finish their benchmark
	benchmark_scores = await asyncio.gather(*benchmark_coros)

	print('sending data to workers')

	# calculates the number of data batches each worker should be assigned
	total_batches = 6000//BATCH_SIZE
	normalizing_factor = total_batches/sum(benchmark_scores)
	data_allocation = [round(normalizing_factor*b) for b in benchmark_scores]

	# assigns data to each worker
	data_allocation_coros = []
	for index, w in enumerate(workers):
		num_batches = data_allocation[index]

		# FUTURE REFERENCE - Additional optimization code will go here
		if wages_per_batch[index] < job_max_price_per_batch:
			# If wage is acceptable, assign the appropriate number of batches and charge the "wallet"
			# gets a bunch of random indices of data samples
			indices = torch.randperm(x_train.shape[0])[0: num_batches*BATCH_SIZE]
			paid_out_for_job[index] = wages_per_batch[index]*num_batches
		else:
			# If wage is too much, do not assign any data
			indices = torch.randperm(x_train.shape[0])[0: 0]

		x_data = x_train[indices]
		y_data = y_train[indices]

		data_allocation_coros.append(w.rpcs.set_training_data(x_data, y_data))

	# waits for data to be sent to workers
	await asyncio.gather(*data_allocation_coros)

	# evaluate parameters
	loss, acc = val_evaluation(net, x_test, y_test)
	print('network loss and validation prior to training:', loss, acc)

	start_time = time.time()

	for i in range(num_global_cycles):
		print('training index:', i, 'out of', num_global_cycles)

		# some workers don't have a GPU and the device that a tensor is on will be serialized, so we've gotta move the network to CPU before transmitting parameters to worker
		net.to('cpu')

		# local updates
		local_update_coros = []
		for w in workers:
			local_update_coros.append(w.rpcs.local_update(list(net.parameters())))

		# waits for local updates to complete
		worker_params = await asyncio.gather(*local_update_coros)

		net.to(device)

		# aggregates parameters
		weights = [d/sum(data_allocation) for d in data_allocation]
		new_params = aggregate_parameters(worker_params, weights)

		# sets the central model to the new parameters
		set_parameters(net, new_params)

		# evaluate new parameters
		loss, acc = val_evaluation(net, x_test, y_test)
		print('network loss and validation:', loss, acc)

	elapsed_time = time.time() - start_time

	timing_logs_coros = []
	for w in workers:
		timing_logs_coros.append(w.rpcs.return_and_clear_timing_logs())

	# wait for timing logs to be returned from each worker
	timing_logs = await asyncio.gather(*timing_logs_coros)

	for index, log in enumerate(timing_logs):
		print('worker ' + str(index) + ' computed ' + str(len(log)) + ' batches in ' + str(sum(log)) + 's')
	print('total elapsed time for job completion ' + str(elapsed_time))
	for index, pay in enumerate(paid_out_for_job):
		print('worker ' + str(index) + ' was payed ' + str(pay) + ' for computing ' + str(pay//wages_per_batch[index]) + ' batches')
	print("total paid out: " + str(sum(paid_out_for_job)))

if (__name__ == '__main__'):
	asyncio.run(main())
