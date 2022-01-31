from axon import discovery, worker
from common import TwoNN, set_parameters
from tqdm import tqdm
import asyncio, requests, torch, time, signal

# the ip address of the notice board
nb_ip = '192.168.2.19'

BATCH_SIZE = 32

device = 'cpu'
if torch.cuda.is_available(): device = 'cuda:0'

net = TwoNN()
criterion = torch.nn.CrossEntropyLoss()

x_train = None
y_train = None

# returns an instance of the optimizer we'll use
def get_optimizer(neural_net):
	return torch.optim.Adam([{'params': neural_net.parameters()}], lr=0.0001)

# rpc that sets the data the worker will train on in their local update
@worker.rpc()
def set_training_data(x_data, y_data):
	global x_train, y_train

	x_train = x_data
	y_train = y_data

# rpc that runs benchmark
@worker.rpc(comms_pattern='duplex', executor='Thread')
def benchmark(num_batches):
	print('running benchmark!')
	global net, criterion, device

	net.to(device)
	optimizer = get_optimizer(net)

	# creating random data
	x_benchmark = torch.randn([BATCH_SIZE*num_batches, 784], dtype=torch.float32)
	y_benchmark = torch.ones([BATCH_SIZE*num_batches], dtype=torch.long)

	# we now train the network on this random data and time how long it takes
	start_time = time.time()

	# training the network on random data
	for batch_number in tqdm(range(num_batches)):
		# getting batch
		x_batch = x_benchmark[batch_number*BATCH_SIZE: (batch_number+1)*BATCH_SIZE].to(device)
		y_batch = y_benchmark[batch_number*BATCH_SIZE: (batch_number+1)*BATCH_SIZE].to(device)

		# getting network's loss on batch
		y_hat = net(x_batch)
		loss = criterion(y_hat, y_batch)

		# updating parameters
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	end_time = time.time()

	# calcuating the training rate of the worker
	batches_per_second = num_batches/(end_time - start_time)

	return batches_per_second

# rpc that performs local update
@worker.rpc(comms_pattern='duplex', executor='Thread')
def local_update(central_model_params):
	print('performing local update')
	global net, criterion, device

	# setting parameters
	set_parameters(net, central_model_params)

	net.to(device)
	optimizer = get_optimizer(net)

	num_training_batches = x_train.shape[0]//BATCH_SIZE

	# training the network on random data
	for batch_number in tqdm(range(num_training_batches)):
		# getting batch
		x_batch = x_train[batch_number*BATCH_SIZE: (batch_number+1)*BATCH_SIZE].to(device)
		y_batch = y_train[batch_number*BATCH_SIZE: (batch_number+1)*BATCH_SIZE].to(device)

		# getting network's loss on batch
		y_hat = net(x_batch)
		loss = criterion(y_hat, y_batch)

		# updating parameters
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return list(net.parameters())

def shutdown_handler(a, b):
	discovery.sign_out(ip=nb_ip)
	exit()

def main():
	# signs into notice board
	discovery.sign_in(ip=nb_ip)

	# registers sign out on sigint
	signal.signal(signal.SIGINT, shutdown_handler)

	# starts the worker, this line blocks
	worker.init()

if (__name__ == '__main__'):
	main()