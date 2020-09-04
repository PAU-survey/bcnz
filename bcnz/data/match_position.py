#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree

def match_position(parent_cat, match_to, max_dist=0.9, drop_duplicates=True):
    """Code for creating the match. Returns a new catalog with the
       matched entries and a new index.
    """

    inst = KDTree(parent_cat[['ra', 'dec']])
    dist, ind = inst.query(match_to[['ra', 'dec']])

    index_name = parent_cat.index.name
    mapping = pd.DataFrame({'dist': dist[:,0], \
                            index_name: parent_cat.index[ind[:,0]], \
                            'match_id': match_to.index})
    mapping = mapping.set_index(index_name)

    # Max separation in degrees.
    max_dist = max_dist / 3600.
    mapping = mapping[mapping.dist < max_dist]

    # A bit convoluted, but needed since the merge is giving problems
    # with hirarchical indexes.
    match_to = match_to.loc[mapping.match_id]
    match_to.loc[mapping.match_id, 'ref_id'] = mapping.index
    match_to = match_to.set_index('ref_id')

    # Double matches (found one..). Drop both.
    if drop_duplicates:
        match_to = match_to.loc[match_to.index.drop_duplicates(False)]

    return match_to
