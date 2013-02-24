#!/usr/bin/env python
# encoding: UTF8

# Runs multiple catalog files in parallel. In the future it might
# be replaced by another framework.

import os
import pdb
import multiprocessing
import types

import bcnz

def prepare_tasks(conf, zdata):
    """Objects encapsulating the runs."""

    assert not (1 < len(zdata['cat_files']) and \
                isinstance(conf['output'], types.NoneType))

    if conf['output']:
        obs_file = zdata['cat_files'][0]
        ans = [bcnz_std.standard(conf, zdata, obs_file, conf['output'])]
        return ans

    tasks = []
    for obs_file in zdata['cat_files']:
        out_file = '%s.bcnz' % os.path.splitext(obs_file)[0]

        in_iter = bcnz.io.ascii.read_cat(conf, zdata, obs_file)
        out_table = bcnz.io.ascii.write_cat(conf, out_file)

        tasks.append(bcnz.tasks.pzcat(conf, zdata, in_iter, out_table))

    return tasks

def run(task):
    """The pool.map require a function. Defining a closure inside
       run_tasks or using lambda fails badly.
    """

    task.run()

def run_tasks(conf, zdata, tasks):
    """Execute either using multiprocessing or just run tasks in serial."""

#    def run(task):
#        task.run()

    use_par = conf['use_par'] and 1 < len(zdata['cat_files'])
    if use_par:
        nthr = conf['nthr']
        ncpu = multiprocessing.cpu_count()
        nparts = nthr if nthr else ncpu
        #pdb.set_trace()
        pool = multiprocessing.Pool(processes=nparts)
        pool.map(run, tasks)
    else:
        for task in tasks:
            task.run()

class pzcat_local:
    def __init__(self, myconf):
        self.conf = bcnz.libconf(myconf)

    def run(self):
        # Estimate the photoz
        zdata = bcnz.zdata.zdata(self.conf)
        zdata= bcnz.model.add_model(self.conf, zdata)

        tasks = prepare_tasks(self.conf, zdata)
        run_tasks(self.conf, zdata, tasks)
