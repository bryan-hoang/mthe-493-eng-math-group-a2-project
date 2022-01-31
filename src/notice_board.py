import axon
import socket

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
print("Notice board hosted at " + local_ip)

nb = axon.discovery.NoticeBoard()

nb.start()