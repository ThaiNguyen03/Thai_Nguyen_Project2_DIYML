import docker
import os
import time
import requests


client = docker.from_env()

container = client.containers.run('login-api', ports={'5000/tcp': 5000}, detach=True)

time.sleep(5)
container.reload()
# Check if the container is running
if container.status == 'running':
    print('Container is running')
else:
    print('Container is not running')


time.sleep(5)
container.stop()
