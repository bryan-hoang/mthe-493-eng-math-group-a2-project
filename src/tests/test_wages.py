from axon import discovery, client
import asyncio

nb_ip = '192.168.56.1'

async def main():
	# starts the RVL
	await client.start_client()

	# find and connect to workers
	worker_ips = discovery.get_ips(ip=nb_ip)

	# instantiates remote worker objects, with which we can call rpcs on each worker
	workers = [client.RemoteWorker(ip) for ip in worker_ips]

	print('setting worker wages')

	# set wages in each worker
	minwage_coros = []
	for index, w in enumerate(workers):
		minwage_coros.append(w.rpcs.set_minimum_wage((index+1)*10))

	# wait for each worker to finish setting their wages
	await asyncio.gather(*minwage_coros)

	print('getting worker wages')

    # get wages for each worker
	minwage_ret_coros = []
	for w in workers:
		minwage_ret_coros.append(w.rpcs.get_minimum_wage())
    
    # wait for each worker to return wages
	wages = await asyncio.gather(*minwage_ret_coros)

	for index, w in enumerate(workers):
		print("worker " + str(worker_ips[index]) + " has wage " + str(wages[index]))

if (__name__ == '__main__'):
	asyncio.run(main())