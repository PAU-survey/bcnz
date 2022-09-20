import os
import numpy as np
import pandas as pd
import psycopg2


def query_alarcon2020(engine):
    """Query for the catalog from Alarcon2020."""

    sql_flux = """
    SELECT flux.*
    FROM pau_cosmos_photoz_v0_4_photometry_c as flux
    """
    cat_flux = pd.read_sql_query(sql_flux, engine)

    sql_photoz = """
    SELECT photoz.ref_id, photoz.i_auto, photoz.zspec_mean
    FROM pau_cosmos_photoz_v0_4_c as photoz
    """
    cat_photoz = pd.read_sql_query(sql_photoz, engine)
    return cat_flux, cat_photoz


def to_dense(cat_in):
    """Convert the input to a dense catalogue."""

    # Makes more sense for how we use the catalogue later
    flux = cat_in.pivot("ref_id", "band", "flux")
    flux_error = cat_in.pivot("ref_id", "band", "flux_error")
    cat = pd.concat({"flux": flux, "flux_error": flux_error}, axis=1)

    return cat


def alarcon2020(engine):

    cat_flux, cat_photoz = query_alarcon2020(engine)

    cat_flux = to_dense(cat_flux)

    breakpoint()
