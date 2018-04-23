#!/usr/bin/env python
# encoding: UTF8

# Default specification of the chunks here. This is separated to
# simplify adding more configurations without making a total
# mess...

import numpy as np
import pandas as pd

def pz_chunks():
    """Specify all the different SED combinations to run over."""

    # Ok, just change this...
    sed_dir = '/home/eriksen/data/photoz/seds/cosmos_noext'

    df = pd.DataFrame()

    X = [(False, ['Ell1_A_0', 'Ell2_A_0', 'Ell3_A_0', 'Ell4_A_0', 'Ell5_A_0','Ell6_A_0']),
         (False, ['Ell6_A_0', 'Ell7_A_0', 'S0_A_0','Sa_A_0','Sb_A_0', 'Sc_A_0']),
         (True, ['Sc_A_0', 'Sd_A_0', 'Sdm_A_0', 'SB0_A_0','SB1_A_0','SB2_A_0']),
         (True, ['SB2_A_0','SB3_A_0','SB4_A_0','SB5_A_0','SB6_A_0','SB7_A_0','SB8_A_0','SB9_A_0','SB10_A_0','SB11_A_0'])]

    # Configurations to use without extinction
    for use_lines, seds in X:
        S = pd.Series({'seds': seds, 'EBV': 0., 'ext_law': 'SB_calzetti',
                       'use_lines': use_lines})
        S['sed_dir'] = sed_dir

        df = df.append(S, ignore_index=True)

    # The star burst configurations...
    with_ext = ['SB4_A_0', 'SB5_A_0','SB6_A_0','SB7_A_0','SB8_A_0','SB9_A_0','SB10_A_0','SB11_A_0']
    laws = ['SB_calzetti', 'SB_calzetti_bump1', 'SB_calzetti_bump2', 'SMC_prevot']
    EBV_values = np.arange(0.05, 0.501, 0.05)

    for ext_law in laws:
        for EBV in EBV_values:
            S = pd.Series({'seds': with_ext, 'EBV': EBV, 'ext_law': ext_law,
                           'use_lines': True})
            S['sed_dir'] = sed_dir
            df = df.append(S, ignore_index=True)

    df['sep_lines'] = 'O'

    return df
