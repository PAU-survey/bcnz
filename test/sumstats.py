import numpy as np


def sigma_general(x, interval=68.0, min_samples=10):
    """
    Calculates the half-interval around the median of a distribution.

    Parameters
    ----------
    x : ndarray of shape (n, )
        Distribution samples.
    interval : float
        Interval around the median.
    min_samples : int
        Function returns nan if len(x)<min_samples
    Returns
    -------
    plow : float
        Returns p[50-interval/2]
    phigh : float
        Returns p[50+interval/2]
    sigma : float
        Returns (phigh - plow) / 2.0
    """
    msg = "Interval needs to be in range [0,100]"
    assert ((interval > 0.0) & (interval < 100.0)), msg

    intervals = [50.0 - interval / 2.0, 50.0 + interval / 2.0]
    if(len(x) > min_samples):
        plow, phigh = np.percentile(x, intervals)
        sigma = (phigh - plow) / 2.0
        return plow, phigh, sigma
    else:
        return np.ones(3)*np.nan


def magbins_sg68_OR_boostrap(dz, mag, magbins, interval=68.0, min_samples=10):
    """
    Calculates half-interval around the median of a distribution conditioned
    on a secondary quantity, and the outlier rate. The outlier rate is defined
    as the tail beyond three intervals around the median. The error is
    calculated by boostrapping subsamples.

    Parameters
    ----------
    dz : ndarray of shape (n, )
        Distribution samples.
    mag : ndarray of shape (n, )
        Secondary quantity to slice the data to compute summary statistics
        from p(dz | mag )
    magbins : ndarray of shape (nbins, )
        Bins used to slice on the secondary quantity.
    interval : float
        Interval around the median for sigma.
    min_samples : int
        Function returns nan if len(x)<min_samples in sigma.
    Returns
    -------
    sg68_boots_mean : ndarray of shape (nbins, )
        Average sigma.
    sg68_boots_std : ndarray of shape (nbins, )
        RMS sigma.
    or_boots_mean : ndarray of shape (nbins, )
        Average outlier rate.
    or_boots_std : ndarray of shape (nbins, )
        RMS outlier rate.
    """
    magbinsc = 0.5*(magbins[1:]+magbins[:-1])
    sg68_boots_mean = np.zeros_like(magbinsc)
    sg68_boots_std = np.zeros_like(magbinsc)
    or_boots_mean = np.zeros_like(magbinsc)
    or_boots_std = np.zeros_like(magbinsc)
    for i in range(len(magbinsc)):
        sel = (mag >= magbins[i]) & (mag < magbins[i+1])
        dz_sub = dz[sel]
        len_sub = len(dz_sub)
        arange_sub = np.arange(len(dz_sub))
        sg68_boots_mag = []
        or_boots_mag = []
        for j in range(1000):
            dz_sub_boots = dz_sub[np.random.choice(arange_sub, len_sub, replace=True)]
            try:
                sg68_boots_mag.append(sigma_general(dz_sub_boots, interval)[-1])
            except:
                sg68_boots_mag.append(np.nan)
            or_boots_mag.append((abs(dz_sub_boots)>3*sg68_boots_mag[j]).sum()/float(len_sub))
        sg68_boots_mean[i] = np.nanmean(sg68_boots_mag)
        sg68_boots_std[i] = np.nanstd(sg68_boots_mag, ddof=1)
        or_boots_mean[i] = np.nanmean(or_boots_mag)
        or_boots_std[i] = np.nanstd(or_boots_mag, ddof=1)

    return sg68_boots_mean, sg68_boots_std, or_boots_mean, or_boots_std


def magbins_sg68_OR_fixedthresh_boostrap(dz, mag, magbins, thresh, interval=68.0, min_samples=10):
    """
    Calculates half-interval around the median of a distribution conditioned
    on a secondary quantity, and the outlier rate. The outlier rate is defined
    as the tail beyond a fixed given interval around the median. The error is
    calculated by boostrapping subsamples.

    Parameters
    ----------
    dz : ndarray of shape (n, )
        Distribution samples.
    mag : ndarray of shape (n, )
        Secondary quantity to slice the data to compute summary statistics
        from p(dz | mag )
    magbins : ndarray of shape (nbins, )
        Bins used to slice on the secondary quantity.
    thresh : float
        Outlier rate fixed threshold interval around the median.
    interval : float
        Interval around the median for sigma.
    min_samples : int
        Function returns nan if len(x)<min_samples in sigma.
    Returns
    -------
    sg68_boots_mean : ndarray of shape (nbins, )
        Average sigma.
    sg68_boots_std : ndarray of shape (nbins, )
        RMS sigma.
    or_boots_mean : ndarray of shape (nbins, )
        Average outlier rate.
    or_boots_std : ndarray of shape (nbins, )
        RMS outlier rate.
    """
    magbinsc = 0.5*(magbins[1:]+magbins[:-1])
    sg68_boots_mean = np.zeros_like(magbinsc)
    sg68_boots_std = np.zeros_like(magbinsc)
    or_boots_mean = np.zeros_like(magbinsc)
    or_boots_std = np.zeros_like(magbinsc)
    for i in range(len(magbinsc)):
        sel = (mag >= magbins[i]) & (mag < magbins[i+1])
        dz_sub = dz[sel]
        len_sub = len(dz_sub)
        arange_sub = np.arange(len(dz_sub))
        sg68_boots_mag = []
        or_boots_mag = []
        for j in range(1000):
            dz_sub_boots = dz_sub[np.random.choice(arange_sub, len_sub, replace=True)]
            sg68_boots_mag.append(sigma_general(dz_sub_boots, interval)[-1])
            or_boots_mag.append((abs(dz_sub_boots) > thresh).sum()/float(len_sub))
        sg68_boots_mean[i] = np.mean(sg68_boots_mag)
        sg68_boots_std[i] = np.std(sg68_boots_mag, ddof=1)
        or_boots_mean[i] = np.mean(or_boots_mag)
        or_boots_std[i] = np.std(or_boots_mag, ddof=1)

    return sg68_boots_mean, sg68_boots_std, or_boots_mean, or_boots_std


def mean_median_boostrap(dz, val, valbins):
    """
    Calculates mean and median of the distribution. The error is
    calculated by boostrapping subsamples.

    Parameters
    ----------
    dz : ndarray of shape (n, )
        Distribution samples.
    val : ndarray of shape (n, )
        Secondary quantity to slice the data to compute summary statistics
        from p(dz | val )
    valbins : ndarray of shape (nbins, )
        Bins used to slice on the secondary quantity.
    Returns
    -------
    mean_boots_mean : ndarray of shape (nbins, )
        Average mean.
    mean_boots_std : ndarray of shape (nbins, )
        RMS mean.
    median_boots_mean : ndarray of shape (nbins, )
        Average median.
    median_boots_std : ndarray of shape (nbins, )
        RMS median.
    """
    valbinsc = 0.5*(valbins[1:]+valbins[:-1])
    mean_boots_mean = np.zeros_like(valbinsc)
    mean_boots_std = np.zeros_like(valbinsc)
    median_boots_mean = np.zeros_like(valbinsc)
    median_boots_std = np.zeros_like(valbinsc)
    for i in range(len(valbinsc)):
        sel = (val >= valbins[i]) & (val < valbins[i+1])
        dz_sub = dz[sel]
        len_sub = len(dz_sub)
        arange_sub = np.arange(len(dz_sub))
        mean_boots_mag = []
        median_boots_mag = []
        for j in range(1000):
            dz_sub_boots = dz_sub[np.random.choice(arange_sub, len_sub, replace=True)]
            mean_boots_mag.append(np.mean(dz_sub_boots))
            median_boots_mag.append(np.median(dz_sub_boots))
        mean_boots_mean[i] = np.mean(mean_boots_mag)
        mean_boots_std[i] = np.std(mean_boots_mag, ddof=1)
        median_boots_mean[i] = np.mean(median_boots_mag)
        median_boots_std[i] = np.std(median_boots_mag, ddof=1)

    return mean_boots_mean, mean_boots_std, median_boots_mean, median_boots_std


def deltaz(zs, zb):
    dz = (zb-zs)/(1.0+zs)
    return dz
