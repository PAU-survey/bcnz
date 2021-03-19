#!/usr/bin/env python
# encoding: UTF8

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

import numpy as np
import pandas as pd
from pathlib import Path
import bcnz

old_filters = [
    "NB_455",
    "NB_465",
    "NB_475",
    "NB_485",
    "NB_495",
    "NB_505",
    "NB_515",
    "NB_525",
    "NB_535",
    "NB_545",
    "NB_555",
    "NB_565",
    "NB_575",
    "NB_585",
    "NB_595",
    "NB_605",
    "NB_615",
    "NB_625",
    "NB_635",
    "NB_645",
    "NB_655",
    "NB_665",
    "NB_675",
    "NB_685",
    "NB_695",
    "NB_705",
    "NB_715",
    "NB_725",
    "NB_735",
    "NB_745",
    "NB_755",
    "NB_765",
    "NB_775",
    "NB_785",
    "NB_795",
    "NB_805",
    "NB_815",
    "NB_825",
    "NB_835",
    "NB_845",
    "galex2500_nuv",
    "u_cfht",
    "B_Subaru",
    "V_Subaru",
    "r_Subaru",
    "i_Subaru",
    "suprime_FDCCD_z",
    "yHSC",
    "Y_uv",
    "J_uv",
    "H_uv",
    "K_uv",
    "IA427.SuprimeCam",
    "IA464.SuprimeCam",
    "IA484.SuprimeCam",
    "IA505.SuprimeCam",
    "IA527.SuprimeCam",
    "IA574.SuprimeCam",
    "IA624.SuprimeCam",
    "IA679.SuprimeCam",
    "IA709.SuprimeCam",
    "IA738.SuprimeCam",
    "IA767.SuprimeCam",
    "IA827.SuprimeCam",
    "NB711.SuprimeCam",
    "NB816.SuprimeCam",
]


def get_bands():
    """Bands used in fit."""

    # The bands to fit.
    NB = [f"pau_nb{x}" for x in 455 + 10 * np.arange(40)]
    BB = [
        "galex2500_nuv",
        "u_cfht",
        "B_Subaru",
        "V_Subaru",
        "r_Subaru",
        "i_Subaru",
        "suprime_FDCCD_z",
        "yHSC",
        "Y_uv",
        "J_uv",
        "H_uv",
        "K_uv",
        "IA427.SuprimeCam",
        "IA464.SuprimeCam",
        "IA484.SuprimeCam",
        "IA505.SuprimeCam",
        "IA527.SuprimeCam",
        "IA574.SuprimeCam",
        "IA624.SuprimeCam",
        "IA679.SuprimeCam",
        "IA709.SuprimeCam",
        "IA738.SuprimeCam",
        "IA767.SuprimeCam",
        "IA827.SuprimeCam",
        "NB711.SuprimeCam",
        "NB816.SuprimeCam",
    ]

    return NB + BB


def get_model(model_dir):

    runs = bcnz.config.test_bayevz()
    modelD = bcnz.model.cache_model_bayevz(model_dir, runs)
    model_calib = bcnz.model.cache_model_bayevz_calib(model_dir, runs)

    return runs, modelD, model_calib


def get_data_hack():

    bands = get_bands()
    # HACK:
    path = "/Users/alarcon/Dropbox/paucats/"
    filename = "pau821_70bands_i23_min002_NB2BB_PUBLIC.csv"
    df = pd.read_table(path + filename, sep=",", header=0)
    df = df[(df.zspec > 0.15) & (df.zspec < 0.25)]

    flux_cols = ["flux_" + x for x in old_filters]
    flux_err_cols = ["flux_err_" + x for x in old_filters]

    flux = df.loc[:, flux_cols]
    flux_err = df.loc[:, flux_err_cols]

    flux = flux.rename(columns=lambda x: x.replace("flux_", ""))
    flux_err = flux_err.rename(columns=lambda x: x.replace("flux_err_", ""))

    flux = flux.rename(columns={x: y for (x, y) in zip(old_filters, bands)})
    flux_err = flux_err.rename(columns={x: y for (x, y) in zip(old_filters, bands)})

    sel = flux < -1
    flux[sel] = np.nan
    flux_err[sel] = np.nan
    cat = pd.concat({"flux": flux, "flux_error": flux_err}, axis=1)

    keepcols = ["ref_id", "zspec", "I_auto"]
    for key in keepcols:
        cat[key] = df[key]

    # HACK:
    path1 = "/Users/alarcon/Dropbox/bpzbcnz/integrate_gauss/"
    zp = np.prod(
        np.loadtxt(
            path1
            + "zp_chi2_withlim_5sig_inv_bayesev_data821_MBE_newcalib_ext_225_70bands_onlyzs_min002_NB2BB_DESpriv_EMfixed_nobug_noirach_FOII_priors_allEM_noexpbug_test1"
        ),
        axis=0,
    )
    zp /= zp[44]
    zp = dict(zip(bands, zp))
    zp = pd.Series(zp)

    galcat_inp = bcnz.calib.apply_zp(cat, zp, norm_bb=False)

    return galcat_inp


def run_photoz(model_dir):
    runs, modelD, model_calib = get_model(model_dir)

    galcat = get_data_hack()

    for key in modelD.keys():
        modelD[key] = modelD[key].sel(band=galcat.flux.columns.values)
        modelD[key] = modelD[key].transpose("z", "sed", "band")

    result = bcnz.bayint.photoz_batch(runs, galcat, modelD, model_calib)
    breakpoint()


if __name__ == "__main__":
    model_dir = "/Users/alarcon/Documents/tmp_bcnz/tmp_model/test1/"
    run_photoz(model_dir)
