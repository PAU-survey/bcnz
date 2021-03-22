import numpy as np
import pandas as pd
from numpy.linalg import _umath_linalg
import time
import mc_genz_cython
from bcnz.bayint.bayevz_tools import delta_function, flux2mag, max_OII_luminosity
import cosmolopy.distance as cd

cosmo = {"omega_M_0": 0.25, "omega_lambda_0": 0.75, "h": 0.7}
cosmo = cd.set_omega_k_0(cosmo)


def get_pz(
    z,
    f_obs,
    ef_obs,
    model,
    muv_model_0,
    prior_vol,
    max_OII_lum,
    ref_mag_ind,
    Nmax=2000,
    approx_dist=None,
    best_evid=None,
):
    """
    Compute p(z|t) for one galaxy and one model.
    Args:
       z (np.array): redshift grid to compute probability. shape (nz,)
       f_obs (np.array): flux measurement for one galaxy. shape (nf,)
       ef_obs (np.array): flux error measurement for one galaxy. shape (nf,)
       model (float): Model fluxes. shape (nz,nt,nf)
       muv_model_0 (nd.array): MUV model fluxes at z=0 and EBV=0.
       prior_vol (nd.array): The prior volumes for each model and redshift.
       max_OII_lum (float): maximum luminosity prior
       ref_mag_ind (float): reference magnitude index in model array
       Nmax (int): number of integration steps in evidence computation
       approx_dist (float): approximation threshold distance
       best_evid (float): current best evidence value
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

    delta1, delta2 = delta_function(model, DM, max_OII_lum, ref_mag_ind)

    a = delta1 - mu
    b = delta2 - mu

    expBaBs = np.einsum("ij,ij->i", mu, Bs)
    detAs = _umath_linalg.det(As, signature="d->d")

    C = np.linalg.cholesky(Ainvs)

    # Integral value for [-oo,+oo]. Upper bound.
    post_inf = (
        0.5 * expBaBs
        - 0.5 * np.log(detAs)
        + (Nmodels / 2.0) * np.log(2 * np.pi)
        - prior_vol
    )

    # Do not run integral if Gaussian is too far away
    # This is a little bit hacked and should consider removing it.
    dists = np.sqrt(
        np.sum(
            np.clip((mu / np.sqrt(np.diagonal(Ainvs, axis1=1, axis2=2))), -np.inf, 0)
            ** 2,
            axis=1,
        )
    )
    sel = dists < approx_dist
    # Don't compute integral if the upper bound is negligible
    # compared to the current best bayesian evidence
    sel = sel & (post_inf > best_evid - np.log(500))

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


def single_photoz(z, f_obs, ef_obs, fmod, fmod_EL, ref_mag_ind, prior_vol):
    """
    Compute p(z,t) for one galaxy.
    Args:
       z (np.array): redshift grid to compute probability. shape (nz,)
       f_obs (np.array): flux measurement for one galaxy. shape (nf,)
       ef_obs (np.array): flux error measurement for one galaxy. shape (nf,)
       fmod (dict): dict with models, each with shape np.array((nz,nt,nf))
       fmod_EL (dict): dict with models for emission line priors
       ref_mag_ind (int): reference magnitude index in model array
       prior_vol (nd.array): The prior volumes for each model and redshift.
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
            prior_vol[it],
            max_OII_lum,
            ref_mag_ind,
            Nmax=1000,
            approx_dist=5.0,
            best_evid=np.max(postinf + np.log(corrs)),
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


def photoz_batch(runs, galcat, fmod, fmod_EL, prior_vol):
    """
    Calculate p(z) for a batch of galaxies.
    Args:
       runs (df): Dataframe contaning the runs metadata.
       galcat (df): Dataframe contaning the galaxy data.
       fmod (dict): dict with models, each with shape np.array((nz,nt,nf))
       fmod_EL (dict): dict with models for emission line priors
       prior_vol (nd.array): The prior volumes for each model and redshift.
    """

    z = runs.loc[0].zgrid
    pz_labels = ["Z%.3f" % x for x in np.round(z, 3)]

    ref_mag = runs.loc[0].ref_mag
    ref_mag_ind = int(np.argwhere(galcat.flux.columns.values == ref_mag))

    post_batch = []
    for index, row in galcat.iterrows():
        f_obs = row.flux.values
        ef_obs = row.flux_error.values

        posterior, max_evidence = single_photoz(
            z, f_obs, ef_obs, fmod, fmod_EL, ref_mag_ind, prior_vol
        )

        posterior = np.einsum("tz->z", posterior)
        S = pd.Series(dict(zip(pz_labels, posterior)))
        S["ref_id"] = int(row.ref_id)
        S["max_evidence"] = max_evidence
        post_batch.append(S)
    post_batch = pd.DataFrame().append(post_batch)
    post_batch.loc[:, "ref_id"] = post_batch.loc[:, "ref_id"].astype(int)

    return post_batch
