#!/usr/bin/env python
# encoding: UTF8

# Default specification of the chunks here. This is separated to
# simplify adding more configurations without making a total
# mess...

from IPython.core import debugger as ipdb
import numpy as np
import pandas as pd

def pz_chunks():
    """Specify all the different SED combinations to run over."""

    sed_dir = '~/data/photoz/seds/cosmos_noext'

    df = pd.DataFrame()
    # Data structure to define how the different chunks should be run. Use:
    # 1 - If including emission lines.
    # 2 - Extinction laws.
    # 3 - The SEDs.

    calzetti = ['SB_calzetti', 'SB_calzetti_bump1', 'SB_calzetti_bump2']
    X = [(False, calzetti, ['Ell1_A_0', 'Ell2_A_0', 'Ell3_A_0', 'Ell4_A_0', 'Ell5_A_0','Ell6_A_0']),
         (False, calzetti, ['Ell6_A_0', 'Ell7_A_0', 'S0_A_0','Sa_A_0','Sb_A_0', 'Sc_A_0']),
         (True, prevot ['Sc_A_0', 'Sd_A_0', 'Sdm_A_0', 'SB0_A_0','SB1_A_0','SB2_A_0']),
         (True, True, ['SB2_A_0','SB3_A_0','SB4_A_0','SB5_A_0','SB6_A_0','SB7_A_0','SB8_A_0','SB9_A_0','SB10_A_0','SB11_A_0']),
         (False, True, ['bc2003_lr_m52_chab_tau03_dust00_age0.509', 'bc2003_lr_m52_chab_tau03_dust00_age8.0', \
                        'bc2003_lr_m62_chab_tau03_dust00_age0.509', 'bc2003_lr_m62_chab_tau03_dust00_age2.1', \
                        'bc2003_lr_m62_chab_tau03_dust00_age2.6', 'bc2003_lr_m62_chab_tau03_dust00_age3.75'])]

    X = pd.DataFrame(X, columns=['use_line', 'use_ext', 'seds'])

    # The values to use for the extinction.
    laws = ['SB_calzetti', 'SB_calzetti_bump1', 'SB_calzetti_bump2', 'SMC_prevot']
    EBV_values = np.arange(0.05, 0.501, 0.05)

    # And then expand this into the actual configuration..
    df = pd.DataFrame()
    for key, row in X.iterrows():
        S = row.copy()
        if not row.use_ext:
            S['EBV'] = 0.
            S['ext_law'] = 'SB_calzetti'
            df = df.append(S, ignore_index=True)

            continue

        for ext_law in laws:
            for EBV in EBV_values:
                S = row.copy()
                S['ext_law'] = ext_law
                S['EBV'] = EBV

                df = df.append(S, ignore_index=True)


    ipdb.set_trace()



    X = [(True, other_seds)]

    # Configurations without without extinction
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

    df['sep_OIII'] = True

    return df
