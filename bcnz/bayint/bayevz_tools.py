import numpy as np
import pandas as pd
from numpy.linalg import _umath_linalg
from pathlib import Path
import bcnz
import time

import mc_genz_cython

import cosmolopy.distance as cd

cosmo = {"omega_M_0": 0.25, "omega_lambda_0": 0.75, "h": 0.7}
cosmo = cd.set_omega_k_0(cosmo)


def flux2mag(f):
    return -2.5 * np.log10(f) - 48.6


def max_OII_luminosity(mean_MU_MR):
    return 2.0 * mean_MU_MR - 54


def delta_function(model, mu, DM, max_OII_lum, ref_mag_ind):
    """Implement the box boundaries of the MCMC for the
    integral computation used in Alarcon+20. There are
    two continuum templates and one emission line template.

    Args:
       model (df): Model fluxes
       mu (np.array): Mean values of the Gaussian likelihood
       DM (np.array): Luminosity distance as a function of redshift
       max_OII_lum (float): maximum luminosity allowed by the luminosity prior
       ref_mag_ind (int): reference magnitude index in model array
    """

    delta1 = np.zeros_like(mu)
    delta2 = np.zeros_like(mu)
    delta2[:, 0] = 10 ** (-0.4 * (-26 + DM + 48.6)) / model[1, 0, ref_mag_ind]
    delta2[:, 1] = 10 ** (-0.4 * (-26 + DM + 48.6)) / model[1, 1, ref_mag_ind]
    delta2[:, 2] = 10 ** (-0.4 * (max_OII_lum + DM))
    return delta1, delta2


def get_pz(
    z,
    f_obs,
    ef_obs,
    model,
    muv_model_0,
    Norm_func,
    max_OII_lum,
    ref_mag_ind,
    Nmax=2000,
    approx=None,
    extra_approx=None,
):
    """
    Compute p(z|t) for one galaxy and one model.
    Args:
       z (np.array): redshift grid to compute probability. shape (nz,)
       f_obs (np.array): flux measurement for one galaxy. shape (nf,)
       ef_obs (np.array): flux error measurement for one galaxy. shape (nf,)
       model (float): Model fluxes. shape (nz,nt,nf)
       muv_model_0 (int): MUV model fluxes at z=0.
       Norm_func (int): reference magnitude index in model array
       max_OII_lum (int): reference magnitude index in model array
       ref_mag_ind (int): reference magnitude index in model array
       Nmax (int): reference magnitude index in model array
       approx (int): reference magnitude index in model array
       extra_approx (int): reference magnitude index in model array
    """

    z_DM = z.copy()
    z_DM = np.where(z_DM < 0.001, 0.001, z_DM,)
    DM = 5 * np.log10(cd.luminosity_distance(z_DM, **cosmo) * 1e5)

    nz, Nmodels, Nbands = model.shape

    # If any of the models has zero flux in all bands at any redshift
    # matrices become non invertible and photoz fails.
    mask = np.sum(np.sum(model == 0, axis=2) == Nbands, axis=1) == 0

    model = model[mask]

    flux = f_obs / (ef_obs)
    Mij = model / ef_obs

    As = np.einsum("ztf,zsf->zts", Mij, Mij)
    Bs = np.einsum("f,ztf->zt", flux, Mij)

    Ainvs = np.linalg.inv(As)
    mu = np.einsum("zi,zij->zj", Bs, Ainvs)

    delta1, delta2 = delta_function(model, mu, DM, max_OII_lum, ref_mag_ind)

    a = delta1 - mu
    b = delta2 - mu

    expBaBs = np.einsum("ij,ij->i", mu, Bs)
    detAs = _umath_linalg.det(As, signature="d->d")

    C = np.linalg.cholesky(Ainvs)

    # Norm = Norm_func(z)
    # Integral upper bound
    post_inf = (
        0.5 * expBaBs - 0.5 * np.log(detAs) + (Nmodels / 2.0) * np.log(2 * np.pi)
    )  # - Norm

    # Don't compute integral if the upper bound is negligible
    dists = np.sqrt(
        np.sum(
            np.clip((mu / np.sqrt(np.diagonal(Ainvs, axis1=1, axis2=2))), -np.inf, 0)
            ** 2,
            axis=1,
        )
    )
    sel = dists < approx

    if extra_approx is not None:
        sel = sel & (post_inf > extra_approx - np.log(500))

    # Compute the correction [0,1] to the integral value.
    corr = np.zeros_like(post_inf)
    corr[sel] = mc_genz_cython.get_posterior_2lims_with_prior_OII_new(
        a[sel], b[sel], C[sel], Nmax, mu[sel], muv_model_0, z[sel], DM[sel]
    )

    corr_output = np.zeros(nz)
    post_inf_output = np.zeros(nz) - 99999999999999.0
    corr_output[mask] = corr
    post_inf_output[mask] = post_inf

    return corr_output, post_inf_output


def single_photoz(z, f_obs, ef_obs, fmod, fmod_EL, ref_mag_ind, prior_norm_model):
    """
    Compute p(z,t) for one galaxy.
    Args:
       z (np.array): redshift grid to compute probability. shape (nz,)
       f_obs (np.array): flux measurement for one galaxy. shape (nf,)
       ef_obs (np.array): flux error measurement for one galaxy. shape (nf,)
       fmod (list): list of models, each with shape np.array((nz,nt,nf))
       fmod_EL (list): list of models for emission line priors
       ref_mag_ind (int): reference magnitude index in model array
       prior_norm_model (list): list of functions predicting
    """

    mask = np.isnan(f_obs)
    f_obs[mask] = 0.0
    ef_obs[mask] = 1e100

    corrs = np.zeros((len(fmod), len(z)))
    postinf = np.zeros((len(fmod), len(z)))

    t0 = time.time()
    for it in fmod.keys():
        mod = fmod[it].values
        mod0_uv = fmod_EL[it].sel(band="galex2500_nuv").sel(z=0).values
        mod0_u = fmod_EL[it].sel(band="u_cfht").sel(z=0).values
        mod0_r = fmod_EL[it].sel(band="r_Subaru").sel(z=0).values

        mean_MUR = np.mean((flux2mag(mod0_u) - flux2mag(mod0_r))[:2])
        max_OII_lum = max_OII_luminosity(mean_MUR)

        pz_result = get_pz(
            z,
            f_obs,
            ef_obs,
            mod,
            mod0_uv,
            prior_norm_model,
            max_OII_lum,
            ref_mag_ind,
            Nmax=1000,
            approx=5.0,
            extra_approx=np.max(postinf + np.log(corrs)),
        )
        corrs[it] = pz_result[0]
        postinf[it] = pz_result[1]
    t1 = time.time()
    print(t1 - t0)

    keep_postinf = postinf.copy()
    max_postinf = np.nanmax(postinf)
    postinf = postinf - max_postinf
    postinf = np.exp(postinf)
    p = postinf * corrs
    p[~np.isfinite(p)] = 0.0
    max_index = np.unravel_index(np.argmax(p), p.shape)
    max_postinf = (
        keep_postinf[max_index[0], max_index[1]]
        + np.log(corrs[max_index[0], max_index[1]])
        - np.sum(np.log(ef_obs))
        - np.sum((f_obs / ef_obs) ** 2) / 2.0
    )

    # p = np.einsum('tz,t->tz', p, priors_lookup[mag_bins_prior_id[ig]])

    return p, max_postinf


def photoz_batch(runs, galcat, modelD, model_calib):

    prior_norm_model = None
    z = runs.loc[0].zgrid
    pz_labels = ["Z%.3f" % x for x in np.round(z, 3)]

    ref_mag = runs.loc[0].ref_mag
    ref_mag_ind = int(np.argwhere(galcat.flux.columns.values == ref_mag))

    post_batch = []
    for index, row in galcat.iterrows():
        f_obs = row.flux.values
        ef_obs = row.flux_error.values

        posterior, max_evidence = single_photoz(
            z, f_obs, ef_obs, modelD, model_calib, ref_mag_ind, prior_norm_model
        )

        posterior = np.einsum("tz->z", posterior)
        S = pd.Series(dict(zip(pz_labels, posterior)))
        S["ref_id"] = int(row.ref_id)
        S["max_evidence"] = max_evidence
        post_batch.append(S)
    post_batch = pd.DataFrame().append(post_batch)
    post_batch.loc[:, "ref_id"] = post_batch.loc[:, "ref_id"].astype(int)

    return post_batch
