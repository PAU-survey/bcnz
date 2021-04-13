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
# encoding: UTF8

# Simple wrapper to cache the output.
from pathlib import Path
import pandas as pd


def cache_zp(output_dir, *args, **kwds):
    """Functionality for caching the zero-points.
       Args:
           run_dir: Directory to store the results.
    """

    output_dir = Path(output_dir)
    path = output_dir / 'zp.h5'


    print('Calibrating the fluxes')

    import bcnz
    if not path.exists():
        zp = bcnz.calib.calib(*args, **kwds)
        zp.to_hdf(path, 'default')
    else:
        zp = pd.read_hdf(path, 'default')

    return zp
