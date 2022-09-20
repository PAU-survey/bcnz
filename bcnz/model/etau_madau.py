import numpy as np


# Also something from Alex...


def etau_madau(wl, z):
    """
    Madau 1995 extinction for a galaxy spectrum at
    (redshift, wavelenght) defined on a grid (z, wl)
    Args:
        wl (np.array): wavelength values of the grid in Angstroms
        z (np.array): redshift values of the grid
    """
    xe = 1.0 + z

    # Madau coefficients
    l = np.array([1216.0, 1026.0, 973.0, 950.0])
    c = np.array([3.6e-3, 1.7e-3, 1.2e-3, 9.3e-4])
    ll = 912.0

    # Lyman series absorption
    Lyman = np.outer(wl ** 3.46, c / l ** 3.46)
    sel1 = np.divide.outer(wl, np.outer(l, xe)) < 1.0
    tau1 = np.sum(np.einsum("ijk,ij->ijk", sel1, Lyman), axis=1).T

    # Photoelectric absorption
    xc = wl / ll
    xc3 = xc ** 3
    photoel = (
        0.25 * np.einsum("j,ij->ij", xc3, np.subtract.outer(xe ** 0.46, xc ** 0.46))
        + 9.4
        * np.einsum("j,ij->ij", xc ** 1.5, np.subtract.outer(xe ** 0.18, xc ** 0.18))
        - 0.7
        * np.einsum("i,ij->ij", xc3, np.subtract.outer(xc ** (-1.32), xe ** (-1.32))).T
        - 0.023 * np.subtract.outer(xe ** 1.68, xc ** 1.68)
    )

    sel2 = (np.divide.outer(wl, ll * xe) <= 1.0).T
    tau2 = sel2 * photoel

    # Add all together
    tau = tau1 + tau2
    tau = np.einsum("ij,j->ij", tau, wl >= ll)
    tau = np.clip(tau, 0, 700)
    etau = np.exp(-tau)
    return etau
