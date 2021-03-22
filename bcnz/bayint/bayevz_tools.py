import numpy as np


def flux2mag(f):
    return -2.5 * np.log10(f) - 48.6


def max_OII_luminosity(mean_MU_MR):
    return 2.0 * mean_MU_MR - 54


def delta_function(model, DM, max_OII_lum, ref_mag_ind):
    """Implement the box boundaries of the MCMC for the
    integral computation used in Alarcon+20. There are
    two continuum templates and one emission line template.

    Args:
       model (df): Model fluxes. shape (nz,nt,nf)
       DM (np.array): Luminosity distance as a function of redshift
       max_OII_lum (float): maximum luminosity allowed by the luminosity prior
       ref_mag_ind (int): reference magnitude index in model array
    """
    nz, Nmodels, Nbands = model.shape

    delta1 = np.zeros((nz, Nmodels))
    delta2 = np.zeros((nz, Nmodels))
    delta2[:, 0] = 10 ** (-0.4 * (-26 + DM + 48.6)) / model[1, 0, ref_mag_ind]
    delta2[:, 1] = 10 ** (-0.4 * (-26 + DM + 48.6)) / model[1, 1, ref_mag_ind]
    delta2[:, 2] = 10 ** (-0.4 * (max_OII_lum + DM))
    return delta1, delta2
