import numpy as np
import pandas as pd
from numpy.linalg import _umath_linalg
import time
from tqdm import tqdm
import mc_genz_cython
from pathlib import Path
from scipy.optimize import minimize
from bcnz.bayint.bayevz_tools import flux2mag, max_OII_luminosity
import cosmolopy.distance as cd

cosmo = {"omega_M_0": 0.25, "omega_lambda_0": 0.75, "h": 0.7}
cosmo = cd.set_omega_k_0(cosmo)


def return_ratio_zp_calibration(f_obs_group, ef_obs_group, model_group, Niter):
    length = len(f_obs_group)
    v_output = []
    for gal_id in range(length):
        f_obs, ef_obs, model = (
            f_obs_group[gal_id],
            ef_obs_group[gal_id],
            model_group[gal_id],
        )
        flux = f_obs / (ef_obs)
        Mij = model / ef_obs
        A = np.einsum("tf,sf->ts", Mij, Mij)
        b = np.einsum("f,tf->t", flux, Mij)
        Ap = np.where(A > 0, A, 0)
        v = 100 * np.ones_like(b)
        for i in range(Niter):
            a = np.einsum("st,t->s", Ap, v)
            m0 = b / a
            vn = m0 * v
            v = vn
        v_output.append(v)
    return v_output


def delta_function_specz(model, DM, max_OII_lum, ref_mag_ind):
    """Implement the box boundaries of the MCMC for the
    integral computation used in Alarcon+20. There are
    two continuum templates and one emission line template.

    Args:
       model (df): Model fluxes. shape (nz,nt,nf)
       DM (np.array): Luminosity distance as a function of redshift
       max_OII_lum (float): maximum luminosity allowed by the luminosity prior
       ref_mag_ind (int): reference magnitude index in model array
    """
    _, Nmodels, Nbands = model.shape
    nz = len(DM)
    delta1 = np.zeros((nz, Nmodels))
    delta2 = np.zeros((nz, Nmodels))
    delta2[:, 0] = 10 ** (-0.4 * (-26 + DM + 48.6)) / model[1, 0, ref_mag_ind]
    delta2[:, 1] = 10 ** (-0.4 * (-26 + DM + 48.6)) / model[1, 1, ref_mag_ind]
    delta2[:, 2] = 10 ** (-0.4 * (max_OII_lum + DM))
    return delta1, delta2


def get_pz_at_specz(
    zspec,
    zspec_id,
    f_obs,
    ef_obs,
    model,
    muv_model_0,
    prior_vol,
    max_OII_lum,
    ref_mag_ind,
    Nmax=2000,
):
    """
    Compute p(z|t) for one galaxy and one model.
    Args:
       zspec (np.array): spectroscopic redshift. shape (ng,)
       zspec_id (np.array): spec-z id in model redshift grid. shape (ng,)
       f_obs (np.array): flux measurement for one galaxy. shape (ng, nf)
       ef_obs (np.array): flux error measurement for one galaxy. shape (ng, nf)
       model (df): Model fluxes. shape (nz,nt,nf)
       muv_model_0 (nd.array): MUV model fluxes at z=0 and EBV=0.
       prior_vol (nd.array): The prior volumes for each model and redshift.
       max_OII_lum (float): maximum luminosity prior
       ref_mag_ind (float): reference magnitude index in model array
       Nmax (int): number of integration steps in evidence computation
       approx_dist (float): approximation threshold distance
       best_evid (float): current best evidence value
    """

    z_DM = zspec.copy()
    z_DM = np.where(z_DM < 0.001, 0.001, z_DM,)
    DM = 5 * np.log10(cd.luminosity_distance(z_DM, **cosmo) * 1e5)

    model_z = model[zspec_id]
    nz, Nmodels, Nbands = model_z.shape

    # If any of the models has zero flux in all bands at any redshift
    # matrices become non invertible and photoz fails.
    mask = np.sum(np.sum(model_z == 0, axis=2) == Nbands, axis=1) == 0

    f_obs = f_obs[mask]
    ef_obs = ef_obs[mask]
    model_z = model_z[mask]

    flux = f_obs / (ef_obs)
    Mij = model_z / ef_obs[:, None, :]

    As = np.einsum("ztf,zsf->zts", Mij, Mij)
    Bs = np.einsum("zf,ztf->zt", flux, Mij)

    Ainvs = np.linalg.inv(As)
    mu = np.einsum("zi,zij->zj", Bs, Ainvs)

    delta1, delta2 = delta_function_specz(model, DM, max_OII_lum, ref_mag_ind)

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
        - prior_vol[zspec_id]
    )

    # Compute the correction [0,1] to the integral value.
    corr = mc_genz_cython.get_posterior_2lims_with_prior_OII_new_specalibration(
        a, b, C, Nmax, mu, muv_model_0, zspec, DM
    )

    corr_output = np.zeros(nz)
    post_inf_output = np.zeros(nz) - 99999999999999.0
    corr_output[mask] = corr
    post_inf_output[mask] = post_inf

    return corr_output, post_inf_output


def best_model(zspec, zspec_id, f_obs, ef_obs, fmod, fmod_EL, ref_mag_ind, prior_vol):
    """
    Compute the best model fluxes for a group of galaxies. First compute the
    best model based on Bayesian evidence. Then compute the most likely model
    fluxes within that model.
    Args:
       zspec (np.array): spectroscopic redshift. shape (ng,)
       zspec_id (np.array): spec-z id in model redshift grid. shape (ng,)
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

    corrs = np.zeros((len(zspec), len(fmod)))
    postinf = np.zeros((len(zspec), len(fmod)))

    t0 = time.time()
    for it in fmod.keys():
        mod = fmod[it].values
        mod0_uv = fmod_EL[it].sel(band="galex2500_nuv").sel(z=0).values
        mod0_u = fmod_EL[it].sel(band="u_cfht").sel(z=0).values
        mod0_r = fmod_EL[it].sel(band="r_Subaru").sel(z=0).values

        mean_MUR = np.mean((flux2mag(mod0_u) - flux2mag(mod0_r))[:2])
        max_OII_lum = max_OII_luminosity(mean_MUR)

        pz_result = get_pz_at_specz(
            zspec,
            zspec_id,
            f_obs,
            ef_obs,
            mod,
            mod0_uv,
            prior_vol[it],
            max_OII_lum,
            ref_mag_ind,
            Nmax=1000,
        )
        corrs[:, it] = pz_result[0]
        postinf[:, it] = pz_result[1]
    t1 = time.time()
    print(t1 - t0)

    postinf = postinf - np.max(postinf, axis=1)[:, None]
    postinf = np.exp(postinf)
    p = postinf * corrs
    p[~np.isfinite(p)] = 0.0
    # p = np.einsum('tz,t->tz', p, priors_lookup[mag_bins_prior_id[ig]])

    mod = [fmod[x].values[zspec_id[i]] for i, x in enumerate(np.argmax(p, axis=1))]
    ratio = return_ratio_zp_calibration(f_obs, ef_obs, mod, Niter=1000)
    ratio = [np.clip(x, 0, np.inf) for x in ratio]
    Fmod = np.array([np.einsum("tf,t->f", mod[i], ratio[i]) for i in range(len(zspec))])
    return Fmod


def cost(C, f, fe, m):
    return np.sum(((m - C * f) / (C * fe)) ** 2)


def compute_zp(fluxes, fluxes_err, Fmods):

    ng, nf = fluxes.shape
    chi2 = np.sum(((fluxes - Fmods) / fluxes_err) ** 2, axis=1)
    selchi2 = chi2 < nf * 2

    zp = np.zeros(nf)
    for i in range(nf):
        S = fluxes[selchi2, i]
        E = fluxes_err[selchi2, i]
        M = Fmods[selchi2, i]
        sel = ((S - M) / E > -5) & ((S - M) / E < 5)
        S, E, M = S[sel], E[sel], M[sel]
        res = minimize(cost, x0=[1.0,], args=(S, E, M), bounds=((0.5, 1.5),))
        zp[i] = res.x

    return zp


def zero_points(runs, galcat, fmod, fmod_EL, prior_vol, Nrounds, zp_tot=None):
    """
    Calculate zero points for calibration galaxies.
    Args:
       runs (df): Dataframe contaning the runs metadata.
       galcat (df): Dataframe contaning the galaxy data.
       fmod (dict): dict with models, each with shape np.array((nz,nt,nf))
       fmod_EL (dict): dict with models for emission line priors
       prior_vol (nd.array): The prior volumes for each model and redshift.
       Nrounds (int): Number of iterations for zero point calibration.
    """

    z = runs.loc[0].zgrid
    ref_mag = runs.loc[0].ref_mag
    ref_mag_ind = int(np.argwhere(galcat.flux.columns.values == ref_mag))

    f_obs = galcat.flux.values
    ef_obs = galcat.flux_error.values
    zspec = galcat.zs.values

    selection = zspec > 0.0
    assert selection.sum() > 0.0, "No galaxies with spec-z"
    f_obs = f_obs[selection]
    ef_obs = ef_obs[selection]
    zspec = zspec[selection]
    zspec_id = np.argmin(abs(np.add.outer(zspec, -z)), axis=1)

    nf = f_obs.shape[1]
    if zp_tot is None:
        zp_tot = np.ones(nf)

    zp_details = {}
    for i in tqdm(range(Nrounds)):
        Fmods = best_model(
            zspec, zspec_id, f_obs, ef_obs, fmod, fmod_EL, ref_mag_ind, prior_vol
        )
        zp = compute_zp(f_obs, ef_obs, Fmods)

        f_obs = f_obs * zp
        ef_obs = ef_obs * zp

        zp_tot *= zp
        zp_details[i] = zp_tot.copy()

    zp_tot = pd.Series(zp_tot, index=galcat.flux.columns)
    zp_details = pd.DataFrame(zp_details, index=galcat.flux.columns)

    return zp_tot, zp_details


def cache_zp(output_dir, *args, **kwds):
    """Functionality for caching the zero-points.
       Args:
           run_dir: Directory to store the results.
    """

    output_dir = Path(output_dir)
    path = output_dir / "zp.h5"

    print("Calibrating the fluxes")

    if not path.exists():
        zp_tot, zp_details = zero_points(*args, **kwds)
        zp_tot.to_hdf(path, "default")
        zp_details.to_hdf(path, "zp_details")
    else:
        zp_tot = pd.read_hdf(path, "default")

    return zp_tot
