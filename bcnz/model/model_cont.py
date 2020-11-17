# Copyright (C) 2020 Martin B. Eriksen
# This file is part of BCNz <https://github.com/PAU-survey/bcnz>.
#
# BCNz is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BCNz is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BCNz.  If not, see <http://www.gnu.org/licenses/>.
#!/usr/bin/env python
# encoding: UTF8

from __future__ import print_function

from IPython.core import debugger as ipdb
import time
import numpy as np
import pandas as pd

from scipy.interpolate import splrep, splev
from scipy.integrate import trapz, simps
from .etau_madau import etau_madau


def calc_r_const(filters):
    """Normalization factor for each filter."""

    # 2.5*log10(clight_AHz) = 46.19205, which you often see applied to
    # magnitudes.
    clight_AHz = 2.99792458e18

    r_const = pd.Series()
    fL = filters.index.unique()
    for fname in fL:
        sub = filters.loc[fname]
        r_const[fname] = 1.0 / simps(sub.response / sub.lmb, sub.lmb) / clight_AHz

    return r_const


def sed_spls(seds):
    """Create a spline of all the SEDs."""

    sedD = {}
    for sed in seds.index.unique():
        sub_sed = seds.loc[sed]
        spl_sed = splrep(sub_sed.lmb, sub_sed.response)

        sedD[sed] = spl_sed

    return sedD


def calc_ext_spl(ext, config):
    """Spline for the extinction."""

    sub = ext[ext.ext_law == config["ext_law"]]
    ext_spl = splrep(sub.lmb, sub.k)

    return ext_spl


def calc_ab(config, filters, seds, ext, r_const):
    """Estimate the fluxes for all filters and SEDs."""

    # Test for missing SEDs.
    missing_seds = set(config["seds"]) - set(seds.index)
    assert not len(missing_seds), "Missing seds: {}".format(missing_seds)

    sedD = sed_spls(seds)
    ext_spl = calc_ext_spl(ext, config)
    z = np.arange(0.0, config["zmax_ab"], config["dz_ab"])

    # Test...
    df = pd.DataFrame()

    int_method = config["int_method"]
    a = 1.0 / (1 + z)
    for i, band in enumerate(filters.index.unique()):
        print("# band", i, "band", band)

        sub_f = filters.loc[band]

        # Define a higher resolution grid.
        _tmp = sub_f.lmb
        int_dz = config["int_dz"]
        lmb = np.arange(_tmp.min(), _tmp.max(), int_dz)

        tau = etau_madau(lmb, z)

        # Evaluate the filter on this grid.
        spl_f = splrep(sub_f.lmb, sub_f.response)
        y_f = splev(lmb, spl_f, ext=1)

        X = np.outer(a, lmb)

        # Only looping over the configured SEDs.
        for sed in config["seds"]:
            t1 = time.time()

            y_sed = splev(X, sedD[sed])
            k_ext = splev(X, ext_spl)
            EBV = config["EBV"]
            y_ext = 10 ** (-0.4 * EBV * k_ext)

            Y = y_ext * y_sed * y_f * lmb * tau

            if int_method == "simps":
                ans = r_const[band] * simps(Y, lmb, axis=1)
            elif int_method == "sum":
                ans = r_const[band] * int_dz * Y.sum(axis=1)

            # This might be overkill in terms of storage, but information in
            # the columns is a pain..
            part = pd.DataFrame({"z": z, "flux": ans})
            part["band"] = band
            part["sed"] = sed
            part["ext_law"] = config["ext_law"]
            part["EBV"] = EBV

            df = df.append(part, ignore_index=True)

            t2 = time.time()

    return df


def model_cont(
    filters,
    seds_df,
    ext,
    seds,
    ext_law,
    EBV,
    zmax_ab=2.05,
    dz_ab=0.001,
    int_dz=1.0,
    int_method="simps",
):
    """The model fluxes for the continuum
       Args:
           filters (df): Filter transmission curves.
           seds_df (df): Galaxy SEDs.
           ext (df): Extinction curves.
           seds (list): SEDs to use.
           ext_law (str): Extinction law.
           EBV (float): Extinction value.
           zmax_ab (float): Maximum redshift in the flux model.
           dz_ab (float): Redshift resolution in the flux model.
           int_dz (float): Resolution when integration.
           int_method (str): Integration method:

    """

    config = {
        "seds": seds,
        "ext_law": ext_law,
        "EBV": EBV,
        "zmax_ab": zmax_ab,
        "dz_ab": dz_ab,
        "int_dz": int_dz,
        "int_method": int_method,
    }

    r_const = calc_r_const(filters)
    ab = calc_ab(config, filters, seds_df, ext, r_const)

    return ab
