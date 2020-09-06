#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree


def match_position(parent_cat, to_match, max_dist=0.9, drop_duplicates=True):
    """Code for creating the match. Returns a new catalog with the
       matched entries and a new index.

       Args:
           parent_cat (df): The parent catalogue.
           to_match (df): Catalogue to match.
           max_dist (float): Maximum matching distance (arc.sec.)
           drop_duplicated (bool): Drop duplicate matches.
    """

    inst = KDTree(parent_cat[['ra', 'dec']])
    dist, ind = inst.query(to_match[['ra', 'dec']])

    index_name = parent_cat.index.name
    mapping = pd.DataFrame({'dist': dist[:, 0],
                            index_name: parent_cat.index[ind[:, 0]],
                            'match_id': to_match.index})
    mapping = mapping.set_index(index_name)

    # Max separation in degrees.
    max_dist = max_dist / 3600.
    mapping = mapping[mapping.dist < max_dist]

    # A bit convoluted, but needed since the merge is giving problems
    # with hirarchical indexes.
    to_match = to_match.loc[mapping.match_id]
    to_match.loc[mapping.match_id, 'ref_id'] = mapping.index
    to_match = to_match.set_index('ref_id')

    # Double matches (found one..). Drop both.
    if drop_duplicates:
        to_match = to_match.loc[to_match.index.drop_duplicates(False)]

    return to_match
