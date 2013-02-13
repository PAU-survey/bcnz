#!/usr/bin/env python
# encoding: UTF8

# Runs multiple catalog files in parallel. In the future it might
# be replaced by another framework.

import os
import pdb
import multiprocessing
import types

import bcnz_std

def prepare_tasks(conf, zdata):
    """Objects encapsulating the runs."""

    pdb.set_trace()
    assert not (1 < len(zdata['cat_files']) and \
                isinstance(conf['output'], types.NoneType))

    if conf['output']:
        obs_file = zdata['cat_files'][0]
        ans = [bcnz_std.standard(conf, zdata, obs_file, conf['output'])]
        return ans

    ans = []
    for obs_file in conf['cat_files']:
        out_file = '%s.bcnz' % os.path.splitext(obs_file)[0]

        ans.append(\
          bcnz_std.standard(conf, zdata, obs_file, out_file))

#1 < len(conf['obs_files'])
    return ans

def run_tasks(conf, tasks):
    """Execute either using multiprocessing or just run tasks in serial."""

    def run(task):
        task.run()

    use_par = conf['use_par'] and 1 < len(conf['obs_files'])
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

class pzcat:
    def __init__(self, conf, zdata, mode):
        tasks = prepare_tasks(conf, zdata)
        run_tasks(conf, tasks)

class pzcat:
    def __init__(self, myconf):
        self.conf = bcnz.libconf(myconf)

    def run(self):
        # Estimate the photoz
        zdata = bcnz.zdata.zdata(self.conf)
        model = bcnz.model.model(self.conf, zdata)

        tasks = prepare_tasks(conf, zdata)
        run_tasks(conf, tasks)
