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

from IPython.core import debugger as ipdb
from pathlib import Path
import xarray as xr

def model_fname(sed, ext_law, EBV):
    """File name when caching the model."""
    
    fname = '{}:{}:{:.3f}.nc'.format(sed, ext_law, EBV)
    
    return fname

def cache_model(model_dir, runs):
    """Load models if already run, otherwise run one.
       Args:
           cache_dir (str): Directory storing the models.
           runs (df): Which runs to use. See the config directory.
    """

    import bcnz
    if runs is None:
        runs = bcnz.config.eriksen2019()
    # The flattened version used for generating the files.
    runs_flat = runs.explode('seds')
    runs_flat['seds'] = runs_flat.seds.map(lambda x: [x])

    
     # Ensure all models are run.
    model_dir = Path(model_dir)
    for i, (_, row) in enumerate(runs_flat.iterrows()):
        sed = row.seds[0]
        fname = model_fname(sed, row.ext_law, row.EBV)
        path = model_dir / fname
        
        if path.exists():
            continue

        print(f'Running model: {i}')
        model = bcnz.model.model_single(**row)
        model.to_netcdf(path)
        
    print('Loading the models.')    
    D = {}
    for i, row in runs.iterrows():
        L = []
        for j, sed in enumerate(row.seds):
            fname = model_fname(sed, row.ext_law, row.EBV)
            path = model_dir / fname

            # The line of creating a new array is very important. Without
            # this the calibration algorithm became 4.5 times slower.
            f_mod = xr.open_dataset(path).flux
            f_mod = xr.DataArray(f_mod)

            if j:
                f_mod = f_mod.sel(sed=[sed])

            # Store with these entries, but suppress them since
            # they affect calculations.
            f_mod = f_mod.squeeze('EBV')
            f_mod = f_mod.squeeze('ext_law')
            f_mod = f_mod.transpose('z', 'band', 'sed')
            L.append(f_mod)
            
        D[i] = xr.concat(L, dim='sed') 
        
    return D