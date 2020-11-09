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
# Creating the model later used for the photo-z fitting.

# Input data
from .all_filters import all_filters
from .load_filters import load_filters
from .extinction_laigle import extinction_laigle
from .load_seds import load_seds
from .line_ratios import line_ratios

# Core model estimation.
from .model_cont import model_cont
from .model_lines import model_lines

# Adjust and rebin.
from .fmod_adjust import fmod_adjust
from .rebin import rebin

from .cache import cache_model
from .nb2bb import nb2bb


def model_single(seds, ext_law, EBV, sep_OIII, sed_dir, use_lines):
    """Create a single model."""

    seds_cont = load_seds(sed_dir)

    ratios = line_ratios()
    filters = all_filters()
    extinction = extinction_laigle()

    # Continuum and lines.
    model_cont_df = model_cont(
        filters, seds_cont, extinction, seds=seds, EBV=EBV, ext_law=ext_law)
    model_lines_df = model_lines(
        ratios, filters, extinction, EBV=EBV, ext_law=ext_law)

    model_orig = fmod_adjust(model_cont_df, model_lines_df)
    model_binned = rebin(model_orig)

    return model_binned
