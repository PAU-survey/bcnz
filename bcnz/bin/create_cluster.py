#!/usr/bin/env python
# encoding: UTF8

# Starts an dask cluster and returns a prompt so one can
# later adjust the number of workers.
from IPython import embed
import dask_jobqueue

cluster = dask_jobqueue.HTCondorCluster(cores=1, workers=50, memory='8 GB', disk='4 GB')

print('Scheduler:')
print(cluster.scheduler_address)

print('\nDashboard:')
print(cluster.dashboard_link)

cluster.scale_up(20)

print("""
Shell which can be used to control the number of workers. For
example

cluster.scale_up(50)

will launch 50 workers. By default 20 workers have started.

""")


embed(header='')
