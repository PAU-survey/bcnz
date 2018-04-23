#!/usr/bin/env python
# encoding: UTF8

# The input galaxy catalogue.
import xdolphin as xd

def cfht_cat():
    """The CFHT catalogue."""

    j1 = xd.Job('match_parent_cat')
    j1.depend['parent_cat'] = xd.Common('ref_cat')
    j1.depend['to_match'] = xd.Job('cfht_d2_flux')

    return j1

def galcat():
    """Combination of the PAU catalogue with the external data."""

    pau = xd.Job('to_sparse')
    pau.depend['cat'] = xd.Common('coadd')

    cat = xd.Job('join_on_index')
    cat.depend['cfht_cat'] = cfht_cat()
    cat.depend['pau_cat'] = pau
    cat.depend['cosmos_cat'] = xd.Job('cosmos_tweaked')

    return cat
