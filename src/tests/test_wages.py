import asyncio

from axon import client, config, discovery

nb_ip = None


async def test_wages():
    global nb_ip

    # grabs notice board ip for discovery use
    axon_local_ips = await discovery.broadcast_discovery(
        num_hosts=1, port=config.comms_config.notice_board_port
    )

    nb_ip = axon_local_ips.pop()

    # starts the RVL
    await client.start_client()

    # find and connect to workers
    worker_ips = discovery.get_ips(ip=nb_ip)

    # instantiates remote worker objects, with which we can call rpcs on each worker
    workers = [client.RemoteWorker(ip) for ip in worker_ips]

    print("setting worker wages")

    # set wages in each worker
    minwage_coros = []
    for index, w in enumerate(workers):
        minwage_coros.append(w.rpcs.set_minimum_wage((index + 1) * 10))

    # wait for each worker to finish setting their wages
    await asyncio.gather(*minwage_coros)

    print("getting worker wages")

    # get wages for each worker
    minwage_ret_coros = []
    for w in workers:
        minwage_ret_coros.append(w.rpcs.get_minimum_wage())

        # wait for each worker to return wages
    wages = await asyncio.gather(*minwage_ret_coros)

    for index, w in enumerate(workers):
        assert wages[index] == (index + 1) * 10
        print(
            "worker "
            + str(worker_ips[index])
            + " has wage "
            + str(wages[index])
        )


if __name__ == "__main__":
    asyncio.run(test_wages())
