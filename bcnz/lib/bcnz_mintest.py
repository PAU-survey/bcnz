#!/usr/bin/env python
import pdb
import time
import numpy as np
from scipy.optimize import fmin, fmin_powell

def mintest(pb, pb_without, z, z_s, iz_b, it_b):

    ngal, nz, nt = pb.shape
    dims = (nz, nt)

    def f(x,i,A):
        x0 = x.astype(np.int)
        dx = x - x0

        res = A[i,x0[0],x0[1]] + \
              dx[0]*A[i,x0[0]+1,x0[1]] + \
              dx[1]*A[i,x0[0],x0[1]+1]
        
        return res

    z_upd = np.zeros((ngal), dtype=np.int)
    t_upd = np.zeros((ngal), dtype=np.int)

    t1 = time.time() 
    for i in range(ngal):
        x = [iz_b[i], it_b[i]]
        try:
            ans = fmin_powell(f, x, args=(i, -pb_without), \
                              disp=False)
            ans = ans.astype(np.int)
            z_upd[i] = ans[0]
            t_upd[i] = ans[1]

#            pdb.set_trace()
        except IndexError:
            z_upd[i] = iz_b[i]
            t_upd[i] = it_b[i]

#        pdb.set_trace()
    t2 = time.time() 
    print('time', t2-t1)

    r = z[iz_b] - z[z_upd]
#    print(r)
    n = np.sum(0.01 < np.abs(r))
#    print('n', n)

    pdb.set_trace()
    return z_upd, t_upd
