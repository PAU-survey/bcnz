#!/usr/bin/env python
# encoding: UTF8

import numpy as np

conf = {
    'opt': True,
    'others': [],
    'order': ['zb', 'zb_min', 'zb_max', 't_b', 'odds', 'z_ml', 't_ml', 'chi2', 'z_s', 'm_0'],
    'ndesi': 4,
    'm_step': 0.1,
    'output': '',
    # Telescope noise parameters.
    'D_tel': 4.2,
    'aperture': 2.0,
    'scale': 0.27,
    'n_exp': 2.,
    'RN': 5.,

    'trays': 'tray_matrix_42NB.txt',
    'exp_trays': [45,45,45,50,60,70],
    'vign': [1.,0.75, 0.375],
    'sky_spec': 'sky_spectrum.txt',

    'add_noise': False,
    'old_model': False,
    'ab_tmp': 'ab',
    'zmax_ab': 12.,
    'dz_ab': 0.01,
    'prior': 'pau',
    'train': False,
    'norm_flux': False,
    'ab_dir': 'AB',
    'filter_dir': 'FILTER',
    'sed_dir': 'SED',
    'p_min': 0.01,
    'merge_peaks': False,
    'interactive': False,
    'plots': False,
    'probs_lite': True,
    'mag': True,
    'spectra': 'spectra.txt',
    'get_z': True,
    'use_par': True,
    'nthr': None,
    'columns': 'mock.columns',
    'add_spec_prob': False,
    'zc': False,
    'nmax': False,
    'probs': False,
    'probs2': False,
    'convolve_p': False,
    'photo_errors': True,
    'z_thr': 0.,
    'color': False,
    'zp_offsets': 0.,
    'ntypes': [],
    'madau': False,
    'new_ab': False,
    'exclude': [],
    'zmin': 0.01,
    'zmax': 10.,
    'unobs': -99.,
    'undet': 99.,
    'delta_m_0': 0.,
    'n_peaks': 1,
    'verbose': True,
    'min_magerr': 0.001,
    'catalog': None,
    'interp': 0,
    'prior': 'hdfn_gen',
    'dz': 0.01,
    'odds': 0.95,
    'min_rms': 0.05}
