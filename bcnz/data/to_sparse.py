#!/usr/bin/env python
# encoding: UTF8

import pandas as pd

def to_sparse(cat_in):
    """Convert the input to a sparse catalogue."""

    flux = cat_in.pivot('ref_id', 'band', 'flux')
    flux_error = cat_in.pivot('ref_id', 'band', 'flux_error')
    cat = pd.concat({'flux': flux, 'flux_error': flux_error}, axis=1)

    return cat
