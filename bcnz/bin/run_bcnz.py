#!/usr/bin/env python
# encoding: UTF8

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
import fire
import numpy as np
import pandas as pd
from pathlib import Path
import functools

import bcnz
from bcnz.bin import DummyObject, run_photoz_dask

import dask
from dask.distributed import Client
import dask.dataframe as dd

def get_bands(field):
    """Bands used in fit."""

    # The bands to fit.
    NB = [f'pau_nb{x}' for x in 455+10*np.arange(40)]
    if field.lower() == 'cosmos':
        BB = ['cfht_u', 'subaru_b', 'subaru_v',
              'subaru_r', 'subaru_i', 'subaru_z']
    elif field.lower() == 'w2':
        BB = ['kids_u', 'kids_g', 'kids_r',
              'kids_i', 'kids_z', 'vista_y',
              'vista_j', 'vista_h', 'vista_ks']
    else:
        BB = ['cfht_u', 'cfht_g', 'cfht_r', 'cfht_i', 'cfht_z']

    fit_bands = NB + BB

    return fit_bands

def get_input(output_dir, model_dir, memba_prod, field, fit_bands,
              only_specz, coadd_file):
    """Get the input to run the photo-z code."""
   

    path_galcat = output_dir / 'galcat_in.pq'

    # The model.
    runs = bcnz.config.eriksen2019()
    modelD = bcnz.model.cache_model(model_dir, runs)

    if not output_dir.exists():
        output_dir.mkdir()

    # In case it's already estimated.
    if path_galcat.exists():
        # Not actually being used...
        galcat_inp = pd.read_parquet(str(path_galcat))

        return runs, modelD, galcat_inp

    # And then estimate the catalogue.
    engine = bcnz.connect_db()
    galcat_specz = bcnz.data.paus_calib_sample(engine, memba_prod, field, coadd_file=coadd_file)
    zp = bcnz.calib.cache_zp(output_dir, galcat_specz, modelD, fit_bands)

    # This should not be the same. We need to modify this later.
  
 
    if only_specz:
        galcat = galcat_specz
    else: 
        galcat = bcnz.data.paus_main_sample(engine, memba_prod, field, coadd_file=coadd_file)

    # Applying the zero-points.
    norm_filter = bcnz.data.catalogs.rband(field)
    galcat_inp = bcnz.calib.apply_zp(galcat, zp, norm_filter=norm_filter)

    # Temporary hack.... 
    galcat_inp = bcnz.fit.flatten_input(galcat_inp) 
    
    galcat_inp.to_parquet(str(path_galcat))

    return runs, modelD, galcat_inp


def fix_model(modelD, fit_bands):
    # Here only the renaming seems needed.
    new_modelD = {}
    for i, v in modelD.items():
        new_modelD[i] = v.sel(band=fit_bands).rename(sed='model')

    return new_modelD


def validate(output_dir, field):
    """Some very simple validation."""

    # The idea is not to perform all validation here, just some simple tests.
    path_pzcat = output_dir / 'pzcat.pq'
    pzcat = dd.read_parquet(path_pzcat)[['zb', 'qz']].compute()

    engine = bcnz.connect_db()
    if field.lower() == 'cosmos':
        specz = bcnz.specz.zcosmos(engine)
        comb = pzcat.join(specz)
        comb = comb[(comb.I_auto < 22.5) & (comb.r50 > 0)
                    & (3 <= comb.conf) & (comb.conf <= 5)]

    elif field.lower() == 'w1':
        specz = bcnz.specz.vipers(engine)

        # Not actually sure if this later will fail because the magi not
        # being available. If someone cares, please fix.
        comb = pzcat.join(specz)
        comb = comb[comb.magi < 22.5]

    elif field.lower() == 'w2':
        specz = bcnz.specz.sdss_gama(engine)

        comb = pzcat.join(specz)
        comb = comb[comb.magi < 22.5]

    elif field.lower() == 'w3':
        specz = bcnz.specz.deep2(engine)
        comb = pzcat.join(specz)
        comb = comb[comb.magi < 22.5]
    

    comb['dx'] = (comb.zb - comb.zspec) / (1 + comb.zspec)

    # Simple sigma68 statistics.
    def fsig68(X): return 0.5*(X.dx.quantile(0.84) - X.dx.quantile(0.16))

    sub = comb[comb.qz < comb.qz.median()]
    print()
    print('sig68 (all)', fsig68(comb))
    print('sig68 (50%)', fsig68(sub))


def run_photoz(output_dir, model_dir, memba_prod, field, fit_bands=None, only_specz=False, 
               ip_dask=None, coadd_file=None):
    """Run the photo-z over a catalogue in the PAUdm database.

       Args:
           output_dir (str): Directory to store the output.
           model_dir (str): Directory for storing flux models.
           memba_prod (int): MEMBA production to use.
           field (str): The observed field.
           fit_bands (list): Which bands to fit.
           only_specz (bool): Only run photo-z for galaxies with spec-z.
           ip_dask (str): IP for Dask scheduler.
           coadd_file (str): Path to file containing the coadds.
    """

   
    output_dir = Path(output_dir)

    fit_bands = get_bands(field)
    runs, modelD, galcat = get_input(
        output_dir, model_dir, memba_prod, field, fit_bands, only_specz, coadd_file)

    run_photoz_dask(runs, modelD, galcat, output_dir, fit_bands, ip_dask)

    validate(output_dir, field)


if __name__ == '__main__':
    fire.Fire(run_photoz)
