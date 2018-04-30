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


    # The configurations without extinction.
    # 1 - If including extinction lines.
    # 2 - SEDs used.
    noext = [(False, ['Ell1_A_0', 'Ell2_A_0', 'Ell3_A_0', 'Ell4_A_0', 'Ell5_A_0','Ell6_A_0']),
             (False, ['Ell6_A_0', 'Ell7_A_0', 'S0_A_0','Sa_A_0','Sb_A_0', 'Sc_A_0']),
             (True, ['Sc_A_0', 'Sd_A_0', 'Sdm_A_0', 'SB0_A_0','SB1_A_0','SB2_A_0']),
             (True, ['SB2_A_0','SB3_A_0','SB4_A_0','SB5_A_0','SB6_A_0','SB7_A_0','SB8_A_0','SB9_A_0','SB10_A_0','SB11_A_0']),
             (False, ['bc2003_lr_m52_chab_tau03_dust00_age0.509', 'bc2003_lr_m52_chab_tau03_dust00_age8.0', \
                      'bc2003_lr_m62_chab_tau03_dust00_age0.509', 'bc2003_lr_m62_chab_tau03_dust00_age2.1', \
                      'bc2003_lr_m62_chab_tau03_dust00_age2.6', 'bc2003_lr_m62_chab_tau03_dust00_age3.75'])]

    # The configurations with the extinction.
    # 1 - If including extinction lines.
    # 2 - The extinction law.
    # 3 - SEDs used.
    sb = ['SB4_A_0','SB5_A_0','SB6_A_0','SB7_A_0','SB8_A_0','SB9_A_0','SB10_A_0','SB11_A_0']
    mix = ['Sc_A_0', 'Sd_A_0', 'Sdm_A_0', 'SB0_A_0', 'SB1_A_0', 'SB2_A_0', 'SB3_A_0', 'SB4_A_0']
    with_ext = [(True, 'SB_calzetti', sb),
                (True, 'SB_calzetti_bump1', sb),
                (True, 'SB_calzetti_bump2', sb),
                (True, 'SMC_prevot', mix)]

    df = pd.DataFrame()
    # Here the no-extinction runs use the Calzetti law with EBV=0.
    for use_lines, seds in noext:
        S = pd.Series({'ext_law': 'SB_calzetti', 'EBV': 0.})
        S['use_lines'] = use_lines
        S['seds'] = seds

        df = df.append(S, ignore_index=True)

    EBV_values = np.arange(0.05, 0.501, 0.05)
    for use_lines, ext_law, seds in with_ext:
        for EBV in EBV_values:
            S = pd.Series({'ext_law': ext_law, 'EBV': EBV})
            S['use_lines'] = use_lines
            S['seds'] = seds

            df = df.append(S, ignore_index=True)

    return df
