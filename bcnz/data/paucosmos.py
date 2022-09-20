import bcnz
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u

# This catalogue is only used by the Bayesian code. One might want to
# integrate the two versions better.

# Something from Alex. To be integrated.

bands = [
    "u",
    "b",
    "v",
    "r",
    "ip",
    "zpp",
    "yhsc",
    "y",
    "j",
    "h",
    "ks",
    "ib427",
    "ib464",
    "ia484",
    "ib505",
    "ia527",
    "ib574",
    "ia624",
    "ia679",
    "ib709",
    "ia738",
    "ia767",
    "ib827",
    "nb711",
    "nb816",
]

band_names = [
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


# From Table 3 in Laigle+16
F_values = {
    "galex2500_nuv": 8.621,
    "u_cfht": 4.660,
    "B_Subaru": 4.020,
    "V_Subaru": 3.117,
    "r_Subaru": 2.660,
    "i_Subaru": 1.991,
    "suprime_FDCCD_z": 1.461,
    "yHSC": 1.298,
    "Y_uv": 1.211,
    "J_uv": 0.871,
    "H_uv": 0.563,
    "K_uv": 0.364,
    "IA427.SuprimeCam": 4.260,
    "IA464.SuprimeCam": 3.843,
    "IA484.SuprimeCam": 3.621,
    "IA505.SuprimeCam": 3.425,
    "IA527.SuprimeCam": 3.264,
    "IA574.SuprimeCam": 2.937,
    "IA624.SuprimeCam": 2.694,
    "IA679.SuprimeCam": 2.430,
    "IA709.SuprimeCam": 2.289,
    "IA738.SuprimeCam": 2.150,
    "IA767.SuprimeCam": 1.996,
    "IA827.SuprimeCam": 1.747,
    "NB711.SuprimeCam": 2.268,
    "NB816.SuprimeCam": 1.787,
}

# Zero-point errors from Alarcon+20
ZP_ERRORS_LAIGLE = {
    "galex2500_nuv": 0.10,
    "u_cfht": 0.02,
    "B_Subaru": 0.02,
    "V_Subaru": 0.02,
    "r_Subaru": 0.02,
    "i_Subaru": 0.02,
    "suprime_FDCCD_z": 0.02,
    "yHSC": 0.02,
    "Y_uv": 0.05,
    "J_uv": 0.05,
    "H_uv": 0.05,
    "K_uv": 0.05,
    "IA427.SuprimeCam": 0.02,
    "IA464.SuprimeCam": 0.02,
    "IA484.SuprimeCam": 0.02,
    "IA505.SuprimeCam": 0.02,
    "IA527.SuprimeCam": 0.02,
    "IA574.SuprimeCam": 0.02,
    "IA624.SuprimeCam": 0.02,
    "IA679.SuprimeCam": 0.02,
    "IA709.SuprimeCam": 0.02,
    "IA738.SuprimeCam": 0.02,
    "IA767.SuprimeCam": 0.02,
    "IA827.SuprimeCam": 0.02,
    "NB711.SuprimeCam": 0.02,
    "NB816.SuprimeCam": 0.02,
}

ZP_ERRORS_PAUS = {f"pau_nb{x}": 0.02 for x in np.arange(455, 850, 10)}


def query_laigle(engine):

    flux_names = ", ".join(["%s_flux_aper3" % x for x in bands])
    fluxerr_names = flux_names.replace("flux", "fluxerr")
    sql = f"""
SELECT number, alpha_j2000, delta_j2000, ebv, flux_galex_nuv,
fluxerr_galex_nuv, {flux_names}, {fluxerr_names},
ip_mag_auto, cm."offset"
FROM cosmos2015_laigle_v1_1 as cm
WHERE ip_mag_auto<24 AND type=0 AND flag_peter=0
    """
    cat = pd.read_sql_query(sql, engine)
    return cat


def query_paus(engine, prod_memba):

    sql = f"""SELECT fac.ref_id, fac.band, fac.flux, fac.flux_error, fac.n_coadd
    FROM forced_aperture_coadd AS fac
    WHERE fac.production_id={prod_memba} AND fac.run=1
    """
    cat = pd.read_sql_query(sql, engine)
    return cat


def query_cosmos(engine):

    sql = """SELECT ra, dec, paudm_id as ref_id
    FROM cosmos"""
    cat = pd.read_sql_query(sql, engine)
    return cat


def unpack_paus_catalog(cat):

    flux = cat.pivot(index="ref_id", columns="band", values="flux")
    flux = flux.rename(
        columns=dict(
            zip(
                flux.columns.values,
                ["flux_pau_nb" + x[2:] for x in flux.columns.values],
            )
        )
    ).reset_index()
    flux_err = cat.pivot(index="ref_id", columns="band", values="flux_error")
    flux_err = flux_err.rename(
        columns=dict(
            zip(
                flux_err.columns.values,
                ["flux_err_pau_nb" + x[2:] for x in flux_err.columns.values],
            )
        )
    ).reset_index()
    n_coadd_tab = cat.pivot(index="ref_id", columns="band", values="n_coadd")
    n_coadd = cat[["n_coadd", "ref_id"]].groupby("ref_id").agg("sum")

    cat = flux.merge(flux_err, on="ref_id")
    cat = cat.merge(n_coadd, on="ref_id")

    return cat, n_coadd_tab


def laigle_rename_columns(cat):

    cat = cat.rename(
        columns={
            "flux_galex_nuv": "flux_galex2500_nuv",
            "fluxerr_galex_nuv": "flux_err_galex2500_nuv",
        }
    )

    for i, x in enumerate(bands):
        cat = cat.rename(
            columns={
                "%s_flux_aper3" % x: "flux_%s" % band_names[i + 1],
                "%s_fluxerr_aper3" % x: "flux_err_%s" % band_names[i + 1],
            }
        )
    return cat


def laigle_apply_ebv_offset(cat):

    cat[cat < -90.0] = np.nan

    _name = "flux_%s" % band_names[0]
    cat.loc[:, _name] *= 10 ** (0.4 * F_values[band_names[0]] * cat["ebv"]) * 1e-29
    _name = "flux_err_%s" % band_names[0]
    cat.loc[:, _name] *= 10 ** (0.4 * F_values[band_names[0]] * cat["ebv"]) * 1e-29

    for x in band_names[1:]:
        _name = "flux_%s" % x
        cat.loc[:, _name] *= 10 ** (-0.4 * cat["offset"])
        cat.loc[:, _name] *= 10 ** (0.4 * F_values[x] * cat["ebv"]) * 1e-29
        _name = "flux_err_%s" % x
        cat.loc[:, _name] *= 10 ** (-0.4 * cat["offset"])
        cat.loc[:, _name] *= 10 ** (0.4 * F_values[x] * cat["ebv"]) * 1e-29
    return cat


def laigle_apply_extra_errors(cat, zp_error=ZP_ERRORS_LAIGLE):

    for x in band_names:
        _namef = "flux_%s" % x
        _namefe = "flux_err_%s" % x

        _flux = cat.loc[:, _namef].values
        _fluxe = cat.loc[:, _namefe].values
        new_fe = np.sqrt(_fluxe ** 2 + (zp_error[x] * abs(_flux)) ** 2)

        cat.loc[:, _namefe] = new_fe

    return cat


def laigle_catalog(engine):

    laiglecat = query_laigle(engine)
    laiglecat = laigle_rename_columns(laiglecat)
    laiglecat = laigle_apply_ebv_offset(laiglecat)

    laiglecat = laigle_apply_extra_errors(laiglecat)

    laiglecat = laiglecat.drop(columns=["offset", "ebv"])

    return laiglecat


def paus_apply_cosmos_scale(cat):

    ab_factor = 10 ** (0.4 * 26)
    cosmos_scale = ab_factor * 10 ** (0.4 * 48.6)

    for x in np.arange(455, 850, 10):
        _name = "flux_pau_nb%s" % x
        cat.loc[:, _name] /= cosmos_scale
        _name = "flux_err_pau_nb%s" % x
        cat.loc[:, _name] /= cosmos_scale
    return cat


def paus_apply_extra_errors(cat, zp_error=ZP_ERRORS_PAUS):

    for x in np.arange(455, 850, 10):
        x = "pau_nb%s" % x

        _namef = "flux_%s" % x
        _namefe = "flux_err_%s" % x

        _flux = cat.loc[:, _namef].values
        _fluxe = cat.loc[:, _namefe].values
        new_fe = np.sqrt(_fluxe ** 2 + (zp_error[x] * abs(_flux)) ** 2)

        cat.loc[:, _namefe] = new_fe
    return cat


def paus_catalog(engine, prod_memba):

    pauscat = query_paus(engine, prod_memba)
    cosmos = query_cosmos(engine)

    pauscat = unpack_paus_catalog(pauscat)[0]
    pauscat = pauscat.merge(cosmos, on="ref_id")

    pauscat = paus_apply_cosmos_scale(pauscat)

    pauscat = paus_apply_extra_errors(pauscat)

    return pauscat


def pausdeep_catalog(engine, prod_memba):

    pauscat = query_paus(engine, prod_memba)
    cosmos = query_cosmos(engine)

    pauscat, n_coadd_tab = unpack_paus_catalog(pauscat)
    pauscat = pauscat.merge(cosmos, on="ref_id")

    pauscat = paus_apply_cosmos_scale(pauscat)
    pauscat = paus_apply_extra_errors(pauscat)

    n_coadd_tab_above = np.sum(n_coadd_tab >= 10, axis=1)
    pauscat = pauscat.loc[n_coadd_tab_above.values >= 30]
    return pauscat


def paus_catalog_matched_pausdeep(engine, prod_memba, prod_memba_deep):

    pausdeep = pausdeep_catalog(engine, prod_memba_deep)
    cosmos = query_cosmos(engine)
    pauscat = query_paus(engine, prod_memba)
    pauscat, n_coadd_tab = unpack_paus_catalog(pauscat)
    pauscat = pauscat.merge(cosmos, on="ref_id")
    pauscat = paus_apply_cosmos_scale(pauscat)
    pauscat = paus_apply_extra_errors(pauscat)

    pauscat = pauscat.loc[pauscat.ref_id.isin(pausdeep.ref_id.values)]
    return pauscat


def match_catalogs(laiglecat, pauscat):

    catalog_laigle = SkyCoord(
        ra=laiglecat["alpha_j2000"].values * u.degree,
        dec=laiglecat["delta_j2000"].values * u.degree,
    )

    catalog_paus = SkyCoord(
        ra=pauscat.ra.values * u.degree, dec=pauscat.dec.values * u.degree
    )

    idx_spec, d2d_spec, d3d = match_coordinates_sky(catalog_laigle, catalog_paus)
    selection = d2d_spec < 1.0 * u.arcsec

    laiglecat = laiglecat.iloc[selection]
    laiglecat.loc[:, "ref_id"] = pauscat.iloc[idx_spec[selection]].ref_id.values

    final_cat = pauscat.merge(laiglecat, on="ref_id")
    return final_cat


def rearrange_flux_columns(catalog):

    f_cols = [
        x
        for x in catalog.columns.values
        if (x.startswith("flux_")) & (~x.startswith("flux_err"))
    ]
    fe_cols = [x for x in catalog.columns.values if x.startswith("flux_err_")]

    flux = catalog.loc[:, f_cols]
    flux_err = catalog.loc[:, fe_cols]

    flux = flux.rename(columns=lambda x: x.replace("flux_", ""))
    flux_err = flux_err.rename(columns=lambda x: x.replace("flux_err_", ""))

    cols = flux.columns.values
    col_name = ["flux"] * len(cols)
    flux = pd.DataFrame(
        flux.values, columns=pd.MultiIndex.from_arrays([col_name, cols])
    )
    col_name = ["flux_error"] * len(cols)
    flux_err = pd.DataFrame(
        flux_err.values, columns=pd.MultiIndex.from_arrays([col_name, cols])
    )

    sel = flux < -1
    flux[sel] = np.nan
    flux_err[sel] = np.nan

    cat = flux.merge(flux_err, left_index=True, right_index=True)

    keepcols = [
        "ref_id",
        "number",
        "alpha_j2000",
        "delta_j2000",
        "ra",
        "dec",
        "ip_mag_auto",
    ]

    for key in keepcols:
        cat[key] = catalog[key]

    return cat


def pauscosmos(engine, prod_memba):

    if prod_memba == 955:
        prod_memba_deep = 984
        pauscat = paus_catalog_matched_pausdeep(engine, prod_memba, prod_memba_deep)
    elif prod_memba == 984:
        pauscat = pausdeep_catalog(engine, prod_memba)
    else:
        raise NotImplementedError("prod_memba needs to be 955 or 984")
    pass

    laiglecat = laigle_catalog(engine)

    final_cat = match_catalogs(laiglecat, pauscat)

    final_cat = rearrange_flux_columns(final_cat)

    return final_cat
