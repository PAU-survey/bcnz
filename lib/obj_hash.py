#!/usr/bin/env python
# encoding: UTF8

from __future__ import print_function, unicode_literals

import sys
import types
import numpy as np

import hashlib

def to_str(x):
    """Convert object x to a string."""

    # Note: This solution is minimal and designet for one 
    #   program specifically. If you try to feed recursive
    #   references or your own objects it will fail! These
    #   parts can of course be created, 

    if isinstance(x, str):
        return x
    elif sys.version_info[0] == 2 and isinstance(x, unicode):
        return x
    elif isinstance(x, int) or \
         isinstance(x, float):
        return str(x)
    elif isinstance(x, types.NoneType):
        return "None" 
    elif isinstance(x, tuple):
        res = []
        for elem in x:
            res.append(to_str(elem))

        return '.'.join(res)
    elif isinstance(x, dict):
        pairs = []
        keys = list(x.keys())
        keys.sort()
        for key in keys:
            value = x[key]

            pairs.append('%s:%s' % (to_str(key), to_str(value)))

        return ','.join(pairs)
    elif isinstance(x, list) or \
         isinstance(x, np.ndarray):
        part = ','.join([to_str(elem) for elem in x])

        return '[%s]' % part
    else:
        print(type(x)) #z
        raise TypeError 

def test_to_str():
    assert to_str('adasdfa') == 'adasdfa'
    assert to_str(('a', 'b', 'c')) == 'a.b.c'
    assert to_str(9) == '9'

def hash_structure(to_hash):
    """Return a hash for to_hash based on our method to_str that
       construct a string from a more complex object.
    """

    if 3 <= sys.version_info[0]:
        to_hash = bytes(to_str(to_hash), 'utf-8')
    else:
        to_hash = bytes(to_str(to_hash))

    a = hashlib.md5(to_hash)

    return a.hexdigest()
