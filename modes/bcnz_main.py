#!/usr/bin/env python
# encoding: UTF8

import os
import pdb
import multiprocessing
import types

import bcnz_std

def prepare_objects(conf, zdata):
    """Objects encapsulating the runs."""

    globals().update(zdata)

#    mstep = .1
#    ninterp = conf['interp']

    assert not (1 < len(conf['obs_files']) and \
                isinstance(conf['output'], types.NoneType))

    if conf['output']:
        obs_file = conf['obs_files'][0]
        ans = [bcnz_std.standard(conf, zdata, obs_file, conf['output'])]
        return ans

    ans = []
    for obs_file in conf['obs_files']:
        out_file = '%s.bpz' % os.path.splitext(obs_file)[0]

        ans.append(\
          bcnz_std.standard(conf, zdata, obs_file, out_file))

#1 < len(conf['obs_files'])
    return ans

def my_f(x):
    x.run_file()

def do_work(conf, ans):

    use_par = conf['use_par'] and 1 < len(conf['obs_files'])
    if use_par:
        nthr = conf['nthr']
        ncpu = multiprocessing.cpu_count()
        nparts = nthr if nthr else ncpu
        #pdb.set_trace()
        pool = multiprocessing.Pool(processes=nparts)
        pool.map(my_f, ans)
    else:
        for x in ans:
            x.run_file()

def wrapper(conf, zdata):
    objs = prepare_objects(conf, zdata)
    do_work(conf, objs)
