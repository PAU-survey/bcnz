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
#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import pandas as pd
import xarray as xr
import numpy as np
from scipy.optimize import curve_fit
from ..model.load_filters import load_filters
from ..model.nb2bb import nb2bb


def funky_hack(config, syn2real, sed, model_norm):
    """Exactly mimic the cuts Alex was making."""

    ratio = syn2real.sel(sed=sed).copy()
    if sed == "OIII":
        ratio[(ratio.z < 0.1) | (0.45 < ratio.z)] = 1.0
    elif sed == "lines":
        flux = model_norm.sel(sed=sed)
        ratio.values[flux < 0.001 * flux.max()] = 1.0

        # Yes, this actually happens in an interval.
        upper = (0.1226,)
        ratio[(ratio.z > 0.1025) & (ratio.z < upper)] = 1.0
    else:
        # The continuum ratios are better behaved.
        pass

    return ratio


def funky_hack_v2(config, syn2real, sed, model_norm):
    """Exactly mimic the new cuts Alex was making."""
    ratio = syn2real.sel(sed=sed).copy()

    # Remove NaNs when flux is zero.
    ratio = ratio.fillna(1.0)

    # Clip within factor 2 ratio. Ratios outside this interval only happen
    # if the model fluxes are very small, in which case the correction
    # is irrelevant, but it can cause numerical problems.
    ratio = ratio.clip(min=0.5, max=2.0)

    assert np.isfinite(ratio).all()

    return ratio


def scale_model(config, coeff, model):
    """Directly scale the model as with the data."""

    # Transform the coefficients.
    norm_band = config["norm_band"]
    coeff = coeff[coeff.bb == norm_band][["nb", "val"]]
    coeff = coeff.set_index(["nb"]).to_xarray().val

    inds = ["band", "z", "sed", "ext_law", "EBV"]
    model = model.set_index(inds)
    model = model.to_xarray().flux
    # A copy is needed for funky_hack..
    model_norm = model.sel(band=norm_band).copy()
    model_NB = model.sel(band=coeff.nb.values)

    # Scaling the fluxes..
    synbb = (model_NB.rename({"band": "nb"}) * coeff).sum(dim="nb")
    syn2real = model_norm / synbb

    for j, xsed in enumerate(model.sed):
        sed = str(xsed.values)
        # syn2real_mod = funky_hack(config, syn2real, sed, model_norm)
        syn2real_mod = funky_hack_v2(config, syn2real, sed, model_norm)

        for i, xband in enumerate(model.band):
            # Here we scale the narrow bands into the broad band system.
            if str(xband.values).startswith("pau_nb"):
                model[i, :, j, :, :] *= syn2real_mod

    # Since we can not directly store xarray.
    model = model.to_dataframe()

    return model


def fmod_adjust(
    model_cont,
    model_lines,
    coeff=False,
    use_lines=True,
    norm_band="",
    scale_synband=False,
):
    """Adjust the model to reflect that the syntetic narrow bands is not
       entirely accurate.

       Args:
           model_cont (df): Continuum model.
           model_lines (df): Emission line model.
           coeff (df): Coeffients for scaling the model.
           use_lines (bool): If including the emission lines.
           norm_band (str): Normalize on this band.
           scale_synband (str): If actually scaling with the band.
    """

    config = {"use_lines": use_lines, "norm_band": norm_band}

    # Here the different parts will have a different zbinning! For the
    # first round I plan adding another layer doing the rebinning. After
    # doing the comparison, we should not rebin at all.

    # Special case of not actually doing anything ...
    if not scale_synband:
        if not use_lines:
            raise Warning(
                "Both 'scale_synband' and 'use_lines' are False. \
                          Combining continuum with lines anyway."
            )
        comb = pd.concat([model_cont, model_lines])
        comb = comb.set_index(["band", "z", "sed", "ext_law", "EBV"])

        return comb

    assert config["norm_band"], "Need to specify the normalization band"
    out_cont = scale_model(config, coeff, model_cont)
    if use_lines:
        out_lines = scale_model(config, coeff, model_lines)
        out = pd.concat([out_cont, out_lines])
    else:
        out = out_cont

    return out


def _line(x, a, b):
    return a * x + b


def scale_data(config, galcat):
    """Scale data."""

    ifnb2bb = config.nb2bb.values[0]

    if not ifnb2bb:
        return galcat
    else:
        norm_band = config["norm_band"].values[0]
        filter_dir = config["filter_dir"].values[0]

        filters = load_filters(filter_dir)

        coeff = nb2bb(filters, norm_band)

        coeff = coeff[coeff.bb == norm_band][["nb", "val"]]
        coeff = coeff.set_index(["nb"]).to_xarray().val

        pau_nb = ["pau_nb%s" % x for x in np.arange(455, 850, 10)]

        pau_syn = galcat.flux.loc[:, pau_nb].values

        pau_syn[pau_syn < 0.0] = np.nan
        miss = np.arange(len(galcat))[np.sum(~np.isnan(pau_syn), axis=1) != 40]

        for ms in miss:
            sel = ~np.isnan(pau_syn[ms])
            popt, pcov = curve_fit(
                _line, np.arange(40)[sel], np.log10(pau_syn[miss[0]][sel])
            )
            pau_syn[ms][~sel] = 10 ** _line(np.arange(40)[~sel], *popt)

        fsyn = np.einsum("ij,j->i", pau_syn, coeff)
        syn2real = galcat.flux[f"{norm_band}"].values / fsyn

        for x in pau_nb:
            _name = ("flux", x)
            galcat.loc[:, _name] *= syn2real
            _name = ("flux_error", x)
            galcat.loc[:, x] *= syn2real

        return galcat
