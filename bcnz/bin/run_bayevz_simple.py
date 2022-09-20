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
import sys

import dask
from dask.distributed import Client
import dask.dataframe as dd

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

old_filters_2 = [
    *["pau_nb%s" % x for x in np.arange(455, 850, 10)],
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

    # runs = bcnz.config.test_bayevz()
    runs = bcnz.config.pauscosmos_deep()

    modelD = bcnz.model.cache_model_bayevz(model_dir, runs)
    model_calib = bcnz.model.cache_model_bayevz_calib(model_dir, runs)

    return runs, modelD, model_calib


def get_data_hack():

    bands = get_bands()
    # HACK:
    path = "/Users/alarcon/Dropbox/paucats/"
    filename = "pau821_70bands_i23_min002_NB2BB_PUBLIC.csv"
    df = pd.read_table(path + filename, sep=",", header=0)
    df = df[(df.zspec > 0.19) & (df.zspec < 0.21)]

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

    galcat_inp = galcat_inp.rename(columns={"zspec": "zs"})
    return galcat_inp


def get_catalog(catalog_name, engine, memba_prod, field, coadd_file):

    if catalog_name == "paus_calib_sample":
        galcat = bcnz.data.paus_calib_sample(
            engine, memba_prod, field, coadd_file=coadd_file
        )
    elif catalog_name == "paus_main_sample":
        galcat = bcnz.data.paus_main_sample(
            engine, memba_prod, field, coadd_file=coadd_file
        )
    elif catalog_name == "pauscosmos":
        galcat = bcnz.data.pauscosmos(engine, memba_prod)

    return galcat


def flatten_input(df):
    """Flattens the input dataframe."""

    flux = df.flux.rename(columns=lambda x: f"flux_{x}")
    flux_error = df.flux_error.rename(columns=lambda x: f"flux_error_{x}")

    comb = pd.concat([flux, flux_error], 1)
    comb["ref_id"] = df["ref_id"]
    return comb


def get_input(
    output_dir,
    model_dir,
    catalog_name,
    memba_prod,
    field,
    fit_bands,
    only_specz,
    coadd_file,
):
    from bcnz.model.fmod_adjust import scale_data

    """Get the input to run the photo-z code."""

    # The model.
    runs, modelD, model_calib = get_model(model_dir)

    # Transpose bands to the same order of the data
    # Transpose columns to the expected order by the code
    for key in modelD.keys():
        modelD[key] = modelD[key].sel(band=get_bands())
        modelD[key] = modelD[key].transpose("z", "sed", "band")

    # Calculate the volume of the prior. Needed for the Bayesian Evidence.
    prior_vol = bcnz.bayint.calculate_prior_volume(
        output_dir, runs, modelD, model_calib, Nsteps=int(1e6)
    )

    if not output_dir.exists():
        output_dir.mkdir()

    path_galcat = output_dir / "galcat_in.pq"
    # In case it's already estimated.
    if path_galcat.exists():
        galcat = pd.read_parquet(str(path_galcat))
        return galcat, runs, modelD, model_calib, prior_vol
    # And then estimate the catalogue.
    engine = bcnz.connect_db_localhost(9000)
    galcat = get_catalog(catalog_name, engine, memba_prod, field, coadd_file)

    # hack
    path = "/Users/alarcon/Dropbox/paucats/"
    filename = "pau821_70bands_i23_min002_NB2BB_PUBLIC.csv"
    df = pd.read_table(path + filename, sep=",", header=0)

    df = df[df.zspec > 0.0][["ref_id", "zspec"]]
    df = pd.DataFrame(
        df.values, columns=pd.MultiIndex.from_arrays([["ref_id", "zs"], ["", ""]])
    )
    galcat = galcat.merge(df, on="ref_id", how="left",)

    # galcat = galcat[(galcat.zs > 0.1) & (galcat.zs < 0.2)]

    galcat = scale_data(runs, galcat)

    # galcat = get_data_hack()

    # Calibrate the zero points.
    zp = bcnz.bayint.cache_zp(
        output_dir, runs, galcat, modelD, model_calib, prior_vol, 100, zp_tot=None
    )
    # Do not normalize the zero points. In principle the zero points are
    # relative, but in practice the prior on p(UV,OII) is absolute,
    # so the absolute zero-points might carry meaning.
    # This also means the data must be calibrated to the AB system.
    # breakpoint()
    galcat = bcnz.calib.apply_zp(galcat, zp, norm_bb=False)

    # Temporary hack....
    galcat = flatten_input(galcat)

    galcat = galcat.iloc[np.arange(10)]
    galcat.to_parquet(str(path_galcat))

    return galcat, runs, modelD, model_calib, prior_vol


def run_photoz_dask(output_dir, model_dir, catalog_name, ip_dask=None):

    raise NotImplementedError("Dask is currently not implemented")
    output_dir = Path(output_dir)

    path_out = Path(output_dir) / "pzcat.pq"
    if path_out.exists():
        print("Photo-z catalogue already exists.")
        return

    memba_prod, field, fit_bands, only_specz, coadd_file = None, None, None, None, None
    galcat, runs, modelD, model_calib, prior_vol = get_input(
        output_dir,
        model_dir,
        catalog_name,
        memba_prod,
        field,
        fit_bands,
        only_specz,
        coadd_file,
    )
    fit_bands = get_bands()
    ref_mag = runs.loc[0].ref_mag
    ref_mag_ind = int(np.argwhere(np.array(fit_bands) == ref_mag))
    config = {
        "zgrid": runs.loc[0].zgrid,
        "fit_bands": fit_bands,
        "ref_mag_ind": ref_mag_ind,
    }
    client = Client(ip_dask) if not ip_dask is None else Client()

    galcat = dd.read_parquet(str(output_dir / "galcat_in.pq"))

    npartitions = 1
    galcat = (
        galcat.reset_index().repartition(npartitions=npartitions).set_index("ref_id")
    )
    pzcat = galcat.map_partitions(
        bcnz.bayint.photoz_dask, config, modelD, model_calib, prior_vol
    )

    pzcat = pzcat.repartition(npartitions=100)
    pzcat = dask.optimize(pzcat)[0]

    pzcat.to_parquet(str(path_out))
    # result = bcnz.bayint.photoz_batch(runs, galcat, modelD, model_calib, prior_vol)


def run_photoz_serial(output_dir, model_dir, catalog_name, memba_prod):

    output_dir = Path(output_dir)

    path_out = Path(output_dir) / "pzcat.pq"
    if path_out.exists():
        print("Photo-z catalogue already exists.")
        return

    field, fit_bands, only_specz, coadd_file = None, None, None, None
    galcat, runs, modelD, model_calib, prior_vol = get_input(
        output_dir,
        model_dir,
        catalog_name,
        memba_prod,
        field,
        fit_bands,
        only_specz,
        coadd_file,
    )
    fit_bands = get_bands()
    ref_mag = runs.loc[0].ref_mag
    ref_mag_ind = int(np.argwhere(np.array(fit_bands) == ref_mag))
    config = {
        "zgrid": runs.loc[0].zgrid,
        "fit_bands": fit_bands,
        "ref_mag_ind": ref_mag_ind,
    }

    pzcat = bcnz.bayint.photoz_batch(galcat, config, modelD, model_calib, prior_vol)

    pzcat.to_parquet(str(path_out))


def _run_photoz_schwimmbad(data):
    galcat, config, modelD, model_calib, prior_vol = data
    pzcat = bcnz.bayint.photoz_batch(galcat, config, modelD, model_calib, prior_vol)
    return pzcat


def run_photoz_schwimmbad(output_dir, model_dir, catalog_name, memba_prod, pool):

    output_dir = Path(output_dir)

    path_out = Path(output_dir) / "pzcat.pq"
    if path_out.exists():
        print("Photo-z catalogue already exists.")
        return

    field, fit_bands, only_specz, coadd_file = None, None, None, None
    galcat, runs, modelD, model_calib, prior_vol = get_input(
        output_dir,
        model_dir,
        catalog_name,
        memba_prod,
        field,
        fit_bands,
        only_specz,
        coadd_file,
    )
    fit_bands = get_bands()
    ref_mag = runs.loc[0].ref_mag
    ref_mag_ind = int(np.argwhere(np.array(fit_bands) == ref_mag))
    config = {
        "zgrid": runs.loc[0].zgrid,
        "fit_bands": fit_bands,
        "ref_mag_ind": ref_mag_ind,
    }

    nranks = pool.comm.Get_size()
    rank_ind = np.array_split(np.arange(len(galcat)), nranks)

    data_batches = []

    for i in range(nranks):
        _data = (galcat.iloc[rank_ind[i]], config, modelD, model_calib, prior_vol)
        data_batches.append(_data)

    result = pool.map(_run_photoz_schwimmbad, data_batches)
    pzcat = pd.concat(result, ignore_index=True)

    pzcat.to_parquet(str(path_out))


if __name__ == "__main__":

    model_dir = "/Users/alarcon/Documents/tmp_bcnz/tmp_model/pauscosmos_test2/"
    # output_dir = "/Users/alarcon/Documents/tmp_bcnz/tmp_output/test1/"
    output_dir = "/Users/alarcon/Documents/tmp_bcnz/tmp_output/pauscosmos_test3/"
    catalog_name = "pauscosmos"
    memba_prod = 984
    # run_photoz(output_dir, model_dir, catalog_name)
    # run_photoz_serial(output_dir, model_dir, catalog_name, memba_prod)

    # """
    from schwimmbad import MPIPool

    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    run_photoz_schwimmbad(output_dir, model_dir, catalog_name, memba_prod, pool)
    # """
