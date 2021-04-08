#!/usr/bin/env python
# encoding: UTF8


from IPython.core import debugger as ipdb
import fire
import numpy as np
import pandas as pd
from pathlib import Path
import functools

import bcnz

import dask
from dask.distributed import Client
import dask.dataframe as dd

import bcnz_paudm

def paus_fromfile(coadds_file,parentcat_file,min_nb=35,
         only_specz=False, secure_spec=False, has_bb=False, sel_gal=True):
    """Load the PAUS data from PAUdm and perform the required
       transformation.

       Args: 
           min_nb (int): Minimum number of narrow bands.
           only_specz (bool): Only selecting galaxy with spectroscopic redshifts.
           secure_spec (bool): Selecting secure spectroscopic redshifts.
           has_bb (bool): Select galaxies with broad bands data.
           sel_gal (bool): Select galaxies.
           coadd_file (str): Path to file containing the coadds.'
           parentcat_file (str): Path to file containing the BB and the zspec.
    """

    import bcnz

    ## loads the nnb photometry 
    paudm_coadd = bcnz.data.load_coadd_file(coadds_file)
    parent_cat =  pd.read_csv(parentcat_file)#.set_index('ref_id')   
    print(paudm_coadd, parent_cat) 

    phot_cols = ['U','B','V','R','I','ZN','DU','DB','DV','DR','DI','DZN']
    cfg = [['U', 'DU', 'cfht_u'],
       ['B', 'DB', 'subaru_b'],
       ['G','DG','cfht_g'],
       ['V', 'DV', 'subaru_v'],
       ['R', 'DR', 'subaru_r'],
       ['I', 'DI', 'subaru_i'],
       ['ZN', 'DZN', 'subaru_z']]

    specz = parent_cat[['ref_id','zspec']]
    flux_cols, err_cols, names = zip(*cfg)
    flux = parent_cat[list(flux_cols)].rename(columns=dict(zip(flux_cols, names)))
    flux_error = parent_cat[list(err_cols)].rename(columns=dict(zip(err_cols, names)))
    parent_cat = pd.concat({'flux': flux, 'flux_error': flux_error}, axis=1)

    data_in = paudm_coadd.join(parent_cat, how='inner')
   

    # Add some minimum noise.
    #data_noisy = data_in
    data_noisy = bcnz.data.fix_noise(data_in)

    # Select a subset of the galaxies.
    conf = {'min_nb': min_nb, 'only_specz': only_specz, 'secure_spec': secure_spec,
            'has_bb': has_bb, 'sel_gal': sel_gal}

    #conf['test_band'] = rband(field)

    nbsubset = bcnz.data.gal_subset(data_noisy, specz, **conf)

    # Synthetic narrow band coefficients.
    #synband = rband(field)
    #filters = bcnz.model.all_filters()
    #coeff = bcnz.model.nb2bb(filters, synband)

    #data_scaled = bcnz.data.synband_scale(nbsubset, coeff, synband=synband,
    #                                      scale_data=True)

    return data_noisy#data_scaled


def get_bands(broad_bands):
    """Bands used in fit."""

    # The bands to fit.
    NB = [f'pau_nb{x}' for x in 455+10*np.arange(40)]
    BB = broad_bands

    fit_bands = NB + BB

    return fit_bands

def get_input(output_dir, model_dir, fit_bands,coadds_file, parentcat_file):
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
    galcat_inp = paus_fromfile(coadds_file=coadds_file,parentcat_file = parentcat_file)
  
    # Temporary hack.... 
    galcat_inp = bcnz.fit.flatten_input(galcat_inp) 
    galcat_inp.to_parquet(str(path_galcat))

    return runs, modelD, galcat_inp


def run_photoz(coadds_file, parentcat_file,output_dir, model_dir, broad_bands= ['cfht_u','subaru_b','subaru_v','subaru_r','subaru_i','subaru_z'], ip_dask=None):
    """Run the photo-z over a external provided catalogue.

       Args:
           output_dir (str): Directory to store the output.
           model_dir (str): Directory for storing flux models. 
           ip_dask (str): IP for Dask scheduler.
           coadds_file (str): Path to file containing the nb fluxes. 
           parentcat_file (str): Path to file containing the bb fluxes and the true redshifts
           broad_bands (list): Which broad bands to fit.
    """
    
    output_dir = Path(output_dir)

    fit_bands = get_bands(broad_bands)
   
    runs, modelD, galcat = get_input(
        output_dir, model_dir, fit_bands, coadds_file, parentcat_file)

    bcnz_paudm.run_photoz_dask(runs, modelD, galcat, output_dir, fit_bands, ip_dask)


if __name__ == '__main__':
    fire.Fire(run_photoz)
