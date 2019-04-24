#!/usr/bin/env python
# encoding: UTF8

# Library containing the methods for estimating photo-z quality
# parameters.

from IPython.core import debugger as ipdb
import numpy as np
import xarray as xr

def zb(pz):
    """The traditional bayesian photo-z."""

    izmin = pz.argmax(dim='z')
    zb = pz.z[izmin]

    return zb

def zb_bpz2(pz):
    """A new photo-z estimate included in BPZ2."""

    zbx = (pz*pz.z).sum(dim='z') / pz.sum(dim='z')

    return zbx

def odds(pz, zb, odds_lim):
    """ODDS quality paramter."""

    # Very manual determination of the ODDS through the
    # cumsum function. xarray is n version 0.9.5 not
    # supporting integration.
    z1 = zb - odds_lim*(1.+zb)
    z2 = zb + odds_lim*(1.+zb)

    # When the galaxy is close to the end of the grid.
    z = pz.z.values
    z1 = np.clip(z1, z[0], z[-1])
    z2 = np.clip(z2, z[0], z[-1])

    # This assumes a regular grid.
    z0 = z[0]
    dz = float(z[1] - z[0])
    bins1 = (z1 - z0) / dz - 1 # Cumsum is estimated at the end
    bins2 = (z2 - z0) / dz - 1
    i1 = np.clip(np.floor(bins1), 0, np.infty).astype(np.int)
    i2 = np.clip(np.floor(bins2), 0, np.infty).astype(np.int)
    db1 = bins1 - i1
    db2 = bins2 - i2

    # Here the cdf is estimated using linear interpolation
    # between the points. This is done because the cdf is
    # changing rapidly for a large sample of galaxies.
    cumsum = pz.cumsum(dim='z')
    E = np.arange(len(pz))

    def C(zbins):
        return cumsum.isel_points(gal=E, z=zbins).values

    cdf1 = db1*C(i1+1) + (1.-db1)*C(i1)
    cdf2 = db2*C(i2+1) + (1.-db2)*C(i2)
    odds = cdf2 - cdf1

    return odds

def pz_width(pz, zb, width_frac):
    """Estimate the pz_width quality parameter."""

    # The redshift width with a fraction frac (see below) of the pdf on
    # either side. The ODDS require a different ODDS limit for different
    # magnitudes and fractions to be optimal, while this quality parameter
    # is more robust. The estimation use the first derivative of the
    # cumsum to estimate pz_width, since a discrete pz_width is problematic
    # when cutting.

    frac = width_frac #0.01
    cumsum = pz.cumsum(dim='z')
    ind1 = (cumsum > frac).argmax(axis=1) - 1
    ind2 = (cumsum > 1-frac).argmax(axis=1) -1

    igal = range(len(cumsum))
    y1_a = cumsum.isel_points(z=ind1, gal=igal)
    dy1 = (cumsum.isel_points(z=ind1+1, gal=igal) - y1_a) / \
          (cumsum.z[ind1+1].values - cumsum.z[ind1].values)

    y2_a = cumsum.isel_points(z=ind2, gal=igal)
    dy2 = (cumsum.isel_points(z=ind2+1, gal=igal) - y2_a) / \
          (cumsum.z[ind2+1].values - cumsum.z[ind2].values)

    dz1 = (frac - y1_a) / dy1
    dz2 = ((1-frac) - y2_a) / dy2
    pz_width = 0.5*(cumsum.z[(cumsum > 1-frac).argmax(axis=1)].values \
                    + dz2.values \
                    - cumsum.z[(cumsum > frac).argmax(axis=1)].values \
                    - dz1.values)

    return pz_width

def Qz(pz, chi2_min, pz_width, zb, odds_lim=0.02):
    """Other quality parameter."""

    odds0p02 = odds(pz, zb, odds_lim)
    Qz_val = chi2_min*pz_width / odds0p02.values

    return Qz_val


# Ok, this should be elsewhere...
def get_arrays(data_df, filters):
    """Read in the arrays and present them as xarrays."""

    raise ValueError('Where is this used??')

    # Seperating this book keeping also makes it simpler to write
    # up different algorithms.
    dims = ('gal', 'band')
    flux = xr.DataArray(data_df['flux'][filters], dims=dims)
    flux_err = xr.DataArray(data_df['flux_err'][filters], dims=dims)

    # Not exacly the best, but Numpy had problems with the fancy
    # indexing.
    to_use = ~np.isnan(flux_err)

    # This gave problems in cases where all measurements was removed..
    flux.values = np.where(to_use, flux.values, 1e-100) #0.) 

    var_inv = 1./(flux_err + 1e-100)**2
    var_inv.values = np.where(to_use, var_inv, 1e-100)
    flux_err.values = np.where(to_use, flux_err, 1e-100)


    return flux, flux_err, var_inv

def get_pzcat(chi2, odds_lim, width_frac):
    """Get photo-z catalogue from the p(z)."""

    pz = np.exp(-0.5*chi2)
    pz_norm = pz.sum(dim=['chunk', 'z'])
    pz_norm = pz_norm.clip(1e-200, np.infty)

    pz = pz / pz_norm
    pz = pz.sum(dim='chunk')

    # Most of this should be moved into the libpzqual
    # library.
    pz = pz.rename({'ref_id': 'gal'})

    zbx = zb(pz).values
    cat = pd.DataFrame()
    cat['zb'] = zbx
    cat['odds'] = odds(pz, zbz, odds_lim).values
    cat['pz_width'] = pz_width(pz, zbz, width_frac).values
    cat['zb_mean'] = zb_bpz2(pz).values

    cat.index = pz.gal.values
    cat.index.name = 'ref_id'

    cat['chi2'] = chi2.min(dim=['chunk', 'z']).sel(ref_id=cat.index)

    # These are now in the "libpzqual" file. I could
    # consider moving them here..
    chi2_min = chi2.min(dim=['chunk', 'z'])
    cat['qual_par'] = (chi2_min*pz_width).values

    odds0p2 = libpzqual.odds(pz, zb, self.config['odds_lim'])
    cat['Qz'] = (chi2_min*pz_width / odds0p2.values).values

    # We need the chunk which contribute most to the redshift
    # peak..
    iz = pz.argmin(dim='z')
    points = chi2.isel_points(ref_id=range(len(chi2.ref_id)), z=iz)
    cat['best_chunk'] = points.argmin(dim='chunk')

    return cat

