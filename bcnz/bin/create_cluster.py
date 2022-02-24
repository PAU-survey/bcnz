#!/usr/bin/env python
# encoding: UTF8

# Copyright (C) 2020 Martin B. Eriksen
# This file is part of BCNz <https://github.com/PAU-survey/bcnz>.
#
# BCNz is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BCNz is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BCNz.  If not, see <http://www.gnu.org/licenses/>.

# Starts an dask cluster and returns a prompt so one can
# later adjust the number of workers.
from IPython import embed
import dask_jobqueue

cluster = dask_jobqueue.HTCondorCluster(cores=1, memory='8 GB', disk='4 GB')

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
