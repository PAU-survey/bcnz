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

def paus_fromfile(mock_cat,bbnaming,bbfit,min_nb=35,
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

 

    paudm_coadd = bcnz.data.load_coadd_file(mock_cat)
    parent_cat =  pd.read_csv(mock_cat)   
    specz = parent_cat[['ref_id','zspec']].drop_duplicates()
    parent_cat = parent_cat[parent_cat.band.isin(bbnaming)]

    


    bbnaming_error = ['D' + e for e in bbnaming]
    parent_cat_df = parent_cat.pivot(index = 'ref_id', columns = 'band', values = 'flux_error').rename(columns=dict(zip(bbnaming,bbnaming_error))).reset_index()
    parent_cat_f = parent_cat.pivot(index = 'ref_id', columns = 'band', values = 'flux').reset_index()
    parent_cat = parent_cat_f.merge(parent_cat_df, on = 'ref_id')
    parent_cat.set_index('ref_id', inplace = True)



    
    list_bands = []
    list_bands.append(bbnaming)
    list_bands.append(bbnaming_error)
    list_bands.append(bbfit)
  
    cfg = np.array(list_bands).T.tolist()

    df_bands = pd.DataFrame(cfg,columns = ['band_name','band_error_name','band'])
  
 
    flux_cols, err_cols, names = zip(*df_bands.values.tolist())

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
    #print(BB,NB)
    fit_bands = NB + BB

    return fit_bands

def get_input(output_dir, model_dir,bbfit, fit_bands,mock_cat,bbnames,calib,field):
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
    galcat_inp = paus_fromfile(mock_cat,bbnames,bbfit)
  
    if calib == True: 
        zp = bcnz.calib.cache_zp(output_dir, galcat_inp, modelD, fit_bands)
        norm_filter = bcnz.data.catalogs.rband(field)
        galcat_inp = bcnz.calib.apply_zp(galcat_inp, zp, norm_filter=norm_filter)

    # Temporary hack.... 
    galcat_inp = bcnz.fit.flatten_input(galcat_inp) 
    galcat_inp.to_parquet(str(path_galcat))

    return runs, modelD, galcat_inp


def run_photoz(mock_file,output_dir, model_dir, bb_fit= ['cfht_u','subaru_b','subaru_v','subaru_r','subaru_i','subaru_z'],bbnames = None, ip_dask=None,calib = False, field = 'COSMOS'):
    """Run the photo-z over a external provided catalogue.

       Args:
           output_dir (str): Directory to store the output.
           model_dir (str): Directory for storing flux models. 
           ip_dask (str): IP for Dask scheduler.
           mock_file (str): Path to file containing the nb fluxes.  
           bb_fit (list): Which broad bands to fit.
           bbnames (list): names of the bb in the input catalogue
    """
  
    output_dir = Path(output_dir)

    fit_bands = get_bands(bb_fit)
   
    runs, modelD, galcat = get_input(
        output_dir, model_dir,bb_fit, fit_bands, mock_file,bbnames, calib,field)

  
    bcnz_paudm.run_photoz_dask(runs, modelD, galcat, output_dir, fit_bands, ip_dask)


if __name__ == '__main__':
    fire.Fire(run_photoz)
