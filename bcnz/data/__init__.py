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
from .cosmos_laigle import cosmos_laigle
from .paudm_coadd import paudm_coadd, load_coadd_file
from .paudm_cosmos import paudm_cosmos
from .paudm_cfhtlens import paudm_cfhtlens

from .match_position import match_position
from .fix_noise import fix_noise

from .gal_subset import gal_subset
from .synband_scale import synband_scale

from . import catalogs
from .catalogs import paus, paus_calib_sample, paus_main_sample
from .paper_catalogs import alarcon2020
