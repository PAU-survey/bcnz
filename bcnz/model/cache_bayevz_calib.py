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
from pathlib import Path
import xarray as xr
from .all_filters import all_filters
from .load_filters import load_filters
from .extinction_laigle import extinction_laigle
from .load_extinction import load_extinction
from .load_seds import load_seds
from .line_ratios import line_ratios
from .etau_madau import etau_madau

# Core model estimation.
from .model_cont import model_cont
from .model_lines import model_lines
from .model_lines_gaussian import model_lines_gaussian

# Adjust and rebin.
from .fmod_adjust import fmod_adjust, scale_model
from .rebin import rebin
from .combine_lines import combine_lines
from .nb2bb import nb2bb

import os
import numpy as np
import pkg_resources


def model_fname(sed, ext_law, EBV):
    """File name when caching the model."""

    fname = "{}:{}:{:.3f}.nc".format(sed, ext_law, EBV)

    return fname


def model_single_cont(seds, ext_law, EBV, zgrid, sed_dir, filter_dir, ext_dir):
    """
    Calculate fluxes for a continuum template.

       Args:
           seds (list): SED names
           ext_law (str): extinction law
           EBV (float): E(B-V) extinction value
           zgrid (np.array): redshift zgrid
           sed_dir (str): SED directory
           filter_dir (str): filter set directory
           ext_dir (str): extinction law directory
    """
    filters = load_filters(filter_dir)
    seds_cont = load_seds(sed_dir)
    extinction = load_extinction(ext_dir, suff="dat")

    model_cont_df = model_cont(
        filters,
        seds_cont,
        extinction,
        seds=seds,
        EBV=EBV,
        ext_law=ext_law,
        zgrid=zgrid,
    )

    inds = ["band", "sed", "ext_law", "EBV", "z"]
    rebinned = model_cont_df.reset_index().set_index(inds)
    model_binned = rebinned.flux.to_xarray()

    return model_binned


def model_single_lines(lines, ext_law, EBV, zgrid, sed_dir, filter_dir, ext_dir):
    """
    Calculate fluxes for a set of emission lines.

       Args:
           lines (dict): line names and line wavelength
           ext_law (str): extinction law
           EBV (float): E(B-V) extinction value
           zgrid (np.array): redshift zgrid
           sed_dir (str): SED directory
           filter_dir (str): filter set directory
           ext_dir (str): extinction law directory
    """
    filters = load_filters(filter_dir)
    extinction = load_extinction(ext_dir, suff="dat")

    model_lines_gauss_df = model_lines_gaussian(
        filters,
        lines,
        extinction,
        EBV=EBV,
        ext_law=ext_law,
        line_width=10,
        line_flux=1e-17,
        zgrid=zgrid,
    )
    inds = ["band", "sed", "ext_law", "EBV", "z"]
    rebinned = model_lines_gauss_df.reset_index().set_index(inds)
    model_binned = rebinned.flux.to_xarray()

    return model_binned


def scale_nb2bb(row, f_mod):
    """
    Scale the narrow bands to the broad bands using
    a synthetic broad band

       Args:
           row (xr): run configuration
           f_mod (xr): model fluxes
    """
    if "nb2bb" in row.index:
        if row.nb2bb:
            f_mod = f_mod.to_dataframe().reset_index()

            filters = load_filters(row.filter_dir)

            coeff = nb2bb(filters, row.norm_band)
            config = {"norm_band": row.norm_band}
            f_mod_scaled = scale_model(config, coeff, f_mod)

            inds = ["band", "sed", "ext_law", "EBV", "z"]
            f_mod_scaled = f_mod_scaled.reset_index().set_index(inds)
            f_mod_scaled = f_mod_scaled.flux.to_xarray()
            return f_mod_scaled

        else:
            return f_mod
    else:
        return f_mod


def cache_model_bayevz_calib(model_dir, runs):
    """Load models if already run, otherwise it runs them.
    These models are used for emission line priors. We need them
    at redshift 0 and EBV=0.
       Args:
           model_dir (str): Directory storing the models.
           runs (df): Which runs to use. See the config directory.
    """

    # filter_dir = "calib_filters/"
    # filter_dir = pkg_resources.resource_filename(__name__, filter_dir)
    zgrid = np.array([0.0])
    EBV = 0.0

    # The flattened version used for generating the files.
    runs_flat = runs.explode("seds")
    runs_flat["seds"] = runs_flat.seds.map(lambda x: [x])

    # Ensure all models are run.
    model_dir = os.path.join(model_dir, "calib/")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_dir = Path(model_dir)
    for i, (_, row) in enumerate(runs_flat.iterrows()):
        sed = row.seds[0]
        fname = model_fname(sed, row.ext_law, row.EBV)
        path = model_dir / fname

        if path.exists():
            continue

        print(f"Running model: {i}")

        model = model_single_cont(
            row.seds,
            row.ext_law,
            EBV,
            zgrid,
            row.sed_dir,
            row.filter_dir_calib,
            row.ext_dir,
        )

        model.to_netcdf(path)

    for i, (_, row) in enumerate(runs.iterrows()):

        if "lines" in row.index:
            fname = model_fname("lines", row.ext_law, row.EBV)
            path = model_dir / fname

            if path.exists():
                continue

            print(f"Running lines model: {i}")
            model = model_single_lines(
                row.lines,
                row.ext_law,
                EBV,
                zgrid,
                row.sed_dir,
                row.filter_dir_calib,
                row.ext_dir,
            )

            model.to_netcdf(path)

    print("Loading the models.")
    D = {}
    for i, row in runs.iterrows():
        L = []
        for sed in row.seds:
            fname = model_fname(sed, row.ext_law, row.EBV)
            path = model_dir / fname

            f_mod = xr.open_dataset(path)

            # Scale the model according to run
            # f_mod = scale_nb2bb(row, f_mod)

            # Keeping this in case it's relevant later.
            # The line of creating a new array is very important. Without
            # this the calibration algorithm became 4.5 times slower.
            # f_mod = xr.open_dataset(path).flux
            # f_mod = xr.DataArray(f_mod)

            # Store with these entries, but suppress them since
            # they affect calculations.
            f_mod = f_mod.squeeze("EBV")
            f_mod = f_mod.squeeze("ext_law")
            f_mod = f_mod.transpose("z", "band", "sed")
            L.append(f_mod.flux)

        if "lines" in row.index:
            fname = model_fname("lines", row.ext_law, row.EBV)
            path = model_dir / fname

            f_mod = xr.open_dataset(path)
            f_mod = f_mod.to_dataframe().reset_index()

            # Lines are previously run, and here can be combined in subgroups
            # according to the configuration
            ratios_keys = [x for x in row.index if x.startswith("line_ratios")]
            for ratio_id, ratio_key in enumerate(ratios_keys):
                f_mod_comb = combine_lines(f_mod.copy(), row[ratio_key])
                f_mod_comb.sed = f"lines_{ratio_id}"
                inds = ["band", "sed", "ext_law", "EBV", "z"]
                f_mod_comb = f_mod_comb.set_index(inds)
                f_mod_comb = f_mod_comb.flux.to_xarray()

                # Scale the model according to run
                # f_mod_comb = scale_nb2bb(row, f_mod_comb)

                f_mod_comb = f_mod_comb.squeeze("EBV")
                f_mod_comb = f_mod_comb.squeeze("ext_law")
                f_mod_comb = f_mod_comb.transpose("z", "band", "sed")
                L.append(f_mod_comb)
        # breakpoint()
        D[i] = xr.concat(L, dim="sed")

    return D
