#!/usr/bin/env python
import pdb
import os
import multiprocessing

import bcnz_std

def find_what(conf, zdata):
    globals().update(zdata)

    mstep = .1

    ninterp = conf['interp']

    ans = []
    for obs_file in conf['obs_files']:
        out_file = '%s.bpz' % os.path.splitext(obs_file)[0]

        ans.append(\
          bcnz_std.standard(conf, zdata, obs_file, out_file, \
                            mstep, ninterp))

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

    objs = find_what(conf, zdata)
    do_work(conf, objs)
