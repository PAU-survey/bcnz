#!/usr/bin/env python
# encoding: UTF8

import math
import sys

def texp(n):
    if n == 0:
        return '1'
    else:
        part = '+ (-pb2)**{0}/{1}.'.format(n, math.factorial(n))

    return texp(n-1)+part

n = sys.argv[1]
res = texp(n)

print(res)
