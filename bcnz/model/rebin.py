# Copyright (C) 2018 Martin B. Eriksen
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
import time
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev, splint


def rebin(model, zmin=0.01, zmax=2.0, dz=0.001):
    """Rebinning the redshift grid of the model.

       Args:
           model (xarray): Flux model.
           zmin (float): Minimum redshift.
           zmax (float): Maximum redshift.
           dz (float): Redshift spacing.
    """

    zgrid = np.arange(zmin, zmax+dz, dz)

    inds = ['band', 'sed', 'ext_law', 'EBV']
    model = model.reset_index().set_index(inds)

    print('starting to rebin')
    t1 = time.time()
    rebinned = pd.DataFrame()
    for key in model.index.unique():
        sub = model.loc[key]
        spl = splrep(sub.z, sub.flux)

        # Just since it failed once..
        try:
            part = pd.DataFrame({'z': zgrid, 'flux': splev(zgrid, spl, ext=2)})
        except ValueError:
            ipdb.set_trace()

        # I needed to set these manually...
        for k1, v1 in zip(model.index.names, key):
            part[k1] = v1

        rebinned = rebinned.append(part)

    print('time', time.time() - t1)
    rebinned = rebinned.reset_index().set_index(inds+['z'])

    # Converting this once and storing as xarray is much more efficient.
    rebinned = rebinned.flux.to_xarray()

    return rebinned
