import os
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import prior_volume_integral
from scipy.stats import norm
from scipy.integrate import quad
from scipy.interpolate import interp1d
from bcnz.bayint.bayevz_tools import delta_function, flux2mag, max_OII_luminosity

from astropy.cosmology import w0waCDM
cosmo = w0waCDM(H0=70, Om0=0.25, Ode0=0.75)

def _calculate_prior_volume(
    zgrid, zid_eval, model, muv_model_0, max_OII_lum, ref_mag_ind, DM, Nsteps
):

    delta1, delta2 = delta_function(model, DM, max_OII_lum, ref_mag_ind)

    Norm = np.zeros(len(zid_eval))
    for i, zid in enumerate(zid_eval):
        range1, range2 = delta1[zid].copy(), delta2[zid].copy()
        range1[range1 < 1e-10] = 1e-10

        prior_mus = delta2[zid, 2:] / 5.0
        prior_sigmas = prior_mus.copy()
        proposal_norm = quad(
            norm.pdf, a=range1[2], b=range2[2], args=((prior_mus[0], prior_sigmas[0]))
        )[0]
        proposal_norm *= (range2[0] - range1[0]) * (range2[1] - range1[1])
        inputs = (
            range1,
            range2,
            zgrid[zid],
            muv_model_0,
            Nsteps,
            prior_mus,
            prior_sigmas,
            proposal_norm,
            DM[zid],
        )
        Norm[i] = prior_volume_integral.integral_MH(*inputs)

    return np.log(Norm)


def calculate_prior_volume(output_dir, runs, fmod, fmod_EL, Nsteps=int(1e6)):
    """
    Calculate the volume of the model prior. The model has two
    continuum templates and one emission line template.
    Args:
       output_dir (str): Output directory path.
       runs (df): Dataframe contaning the runs metadata.
       fmod (dict): dict with models, each with shape np.array((nz,nt,nf))
       fmod_EL (dict): dict with models for emission line priors
       Nsteps (int): Number of integration steps in metropolis-hastings.
    """

    zgrid = runs.loc[0].zgrid
    z_DM = zgrid.copy()
    z_DM = np.where(z_DM < 0.001, 0.001, z_DM,)
    DM = 5 * np.log10(np.array(cosmo.luminosity_distance(z_DM)) * 1e5)

    ref_mag = runs.loc[0].ref_mag
    prior_volume_zid = runs.loc[0].prior_volume_zid
    zgrid_eval = zgrid[prior_volume_zid]

    ref_mag_ind = int(np.argwhere(fmod[0].band.values == ref_mag))

    norms = np.zeros((len(fmod), len(zgrid)))

    output_dir = os.path.join(output_dir, "prior_volume/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = Path(output_dir)

    for it in fmod.keys():
        path = output_dir / f"pvol_model_{it}.nc"
        if path.exists():
            da = xr.open_dataset(path)
            norms[it] = da.logpvol.values
            continue

        print(f"Running prior volume model: {it}")
        mod = fmod[it].values
        mod0_uv = fmod_EL[it].sel(band="galex2500_nuv").sel(z=0).values
        mod0_u = fmod_EL[it].sel(band="u_cfht").sel(z=0).values
        mod0_r = fmod_EL[it].sel(band="r_Subaru").sel(z=0).values

        mean_MUR = np.mean((flux2mag(mod0_u) - flux2mag(mod0_r))[:2])
        max_OII_lum = max_OII_luminosity(mean_MUR)

        lognorm = _calculate_prior_volume(
            zgrid, prior_volume_zid, mod, mod0_uv, max_OII_lum, ref_mag_ind, DM, Nsteps
        )

        extrapol = interp1d(
            zgrid_eval,
            lognorm,
            bounds_error=False,
            fill_value=(lognorm[0], lognorm[-1]),
        )
        norms[it] = extrapol(zgrid)

        da = pd.DataFrame(data=np.c_[zgrid, norms[it]], columns=["z", "logpvol"])
        da = da.set_index("z").logpvol.to_xarray()
        da.to_netcdf(path)

    return norms
