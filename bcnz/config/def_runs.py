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

# Default specification of the chunks here.

from IPython.core import debugger as ipdb
import numpy as np
import pandas as pd


def eriksen2019():
    """The different runs used in eriksen2019."""

    # The configurations without extinction.
    # 1 - If including extinction lines.
    # 2 - SEDs used.
    noext = [
        (
            False,
            ["Ell1_A_0", "Ell2_A_0", "Ell3_A_0", "Ell4_A_0", "Ell5_A_0", "Ell6_A_0"],
        ),
        (False, ["Ell6_A_0", "Ell7_A_0", "S0_A_0", "Sa_A_0", "Sb_A_0", "Sc_A_0"]),
        (True, ["Sc_A_0", "Sd_A_0", "Sdm_A_0", "SB0_A_0", "SB1_A_0", "SB2_A_0"]),
        (
            True,
            [
                "SB2_A_0",
                "SB3_A_0",
                "SB4_A_0",
                "SB5_A_0",
                "SB6_A_0",
                "SB7_A_0",
                "SB8_A_0",
                "SB9_A_0",
                "SB10_A_0",
                "SB11_A_0",
            ],
        ),
        (
            False,
            [
                "bc2003_lr_m52_chab_tau03_dust00_age0.509",
                "bc2003_lr_m52_chab_tau03_dust00_age8.0",
                "bc2003_lr_m62_chab_tau03_dust00_age0.509",
                "bc2003_lr_m62_chab_tau03_dust00_age2.1",
                "bc2003_lr_m62_chab_tau03_dust00_age2.6",
                "bc2003_lr_m62_chab_tau03_dust00_age3.75",
            ],
        ),
    ]

    # The configurations with the extinction.
    # 1 - If including extinction lines.
    # 2 - The extinction law.
    # 3 - SEDs used.
    sb = [
        "SB4_A_0",
        "SB5_A_0",
        "SB6_A_0",
        "SB7_A_0",
        "SB8_A_0",
        "SB9_A_0",
        "SB10_A_0",
        "SB11_A_0",
    ]
    mix = [
        "Sc_A_0",
        "Sd_A_0",
        "Sdm_A_0",
        "SB0_A_0",
        "SB1_A_0",
        "SB2_A_0",
        "SB3_A_0",
        "SB4_A_0",
    ]
    with_ext = [
        (True, "SB_calzetti", sb),
        (True, "SB_calzetti_bump1", sb),
        (True, "SB_calzetti_bump2", sb),
    ]
    #                (True, 'SMC_prevot', mix)]

    L = []
    # Here the no-extinction runs use the Calzetti law with EBV=0.
    for use_lines, seds in noext:
        S = pd.Series({'ext_law': 'SB_calzetti', 'EBV': 0.})
        S['use_lines'] = use_lines
        S['seds'] = seds
        L.append(S)

    EBV_values = np.arange(0.05, 0.501, 0.05)
    for use_lines, ext_law, seds in with_ext:
        for EBV in EBV_values:
            S = pd.Series({'ext_law': ext_law, 'EBV': EBV})
            S['use_lines'] = use_lines
            S['seds'] = seds
            L.append(S)

    df = pd.concat(L, axis=1).T

    # This applies to all configurations.
    sed_dir = "~/data/photoz/seds/cosmos_noext"
    df["sep_OIII"] = True
    df["sed_dir"] = sed_dir

    return df


def test_bayevz():
    """Different runs to prepare for Alarcon2020 Bayevz runs"""

    nb2bb = True
    norm_band = "r_Subaru"
    ref_mag = "i_Subaru"
    zgrid = np.arange(0, 0.3, 0.001)
    prior_volume_zid = np.linspace(0, len(zgrid) - 1, 10).astype(int)

    lines = {
        "OII": 3726.8,
        "OIII_1": 4959.0,
        "OIII_2": 5007.0,
        "Halpha": 6562.8,
        "Hbeta": 4861,
        "Lyalpha": 1215.7,
        "NII_1": 6548.0,
        "NII_2": 6583.0,
        "SII_1": 6716.44,
        "SII_2": 6730.82,
    }

    # Median values from Alarcon2020
    line_ratios = {
        "OII": 1.0,
        "OIII_1": 0.31892333 * (1.0 / 3.0),
        "OIII_2": 0.31892333,
        "Halpha": 0.8227787,
        "Hbeta": 0.27377869,
        "Lyalpha": 2.0,
        "NII_1": 0.8227787 * 0.35 * 0.3,
        "NII_2": 0.8227787 * 0.35,
        "SII_1": 0.8227787 * 0.35,
        "SII_2": 0.8227787 * 0.35,
    }

    noext = [
        ["Ell1_A_0", "Ell4_A_0"],
        ["Ell4_A_0", "Ell7_A_0"],
        ["Ell7_A_0", "Sc_A_0"],
        ["Sc_A_0", "SB0_A_0"],
        ["SB0_A_0", "SB4_A_0"],
        ["SB4_A_0", "SB8_A_0"],
        ["SB8_A_0", "SB11_A_0"],
    ]

    calzetti = [
        ["Sc_A_0", "SB0_A_0"],
        ["SB0_A_0", "SB4_A_0"],
        ["SB4_A_0", "SB8_A_0"],
        ["SB8_A_0", "SB11_A_0"],
    ]

    EBV_values = np.arange(0.05, 0.501, 0.05)
    df = pd.DataFrame()
    ext_law = "none"
    for seds in noext:
        S = pd.Series({"ext_law": ext_law, "EBV": 0.0})
        S["seds"] = seds
        S["lines"] = lines
        S["line_ratios"] = line_ratios
        S["zgrid"] = zgrid
        S["nb2bb"] = nb2bb
        S["norm_band"] = norm_band
        S["ref_mag"] = ref_mag
        S["prior_volume_zid"] = prior_volume_zid
        df = df.append(S, ignore_index=True)

    ext_law = "SB_calzetti_bump2"
    for seds in calzetti:
        for EBV in EBV_values:
            S = pd.Series({"ext_law": ext_law, "EBV": EBV})
            S["seds"] = seds
            S["lines"] = lines
            S["line_ratios"] = line_ratios
            S["zgrid"] = zgrid
            S["nb2bb"] = nb2bb
            S["norm_band"] = norm_band
            S["ref_mag"] = ref_mag
            S["prior_volume_zid"] = prior_volume_zid
            df = df.append(S, ignore_index=True)
    # This applies to all configurations.

    test_data_dir = "/Users/alarcon/Documents/source/bcnz/test/data/"
    filter_dir = test_data_dir + "filt_70bands_noirach/"
    sed_dir = test_data_dir + "seds/"
    ext_dir = test_data_dir + "ext_laws/"
    filter_dir_calib = test_data_dir + "calib_filters/"
    df["sed_dir"] = sed_dir
    df["filter_dir"] = filter_dir
    df["filter_dir_calib"] = filter_dir_calib
    df["ext_dir"] = ext_dir

    return df


def pauscosmos_deep():
    """Different runs to prepare for Alarcon2020 Bayevz runs"""

    nb2bb = True
    norm_band = "r_Subaru"
    ref_mag = "i_Subaru"
    # zgrid = np.arange(0, 0.3, 0.001)
    zgrid = np.concatenate(
        (
            np.arange(0.0, 1.0, 0.001),
            np.arange(1.0, 1.5, 0.002),
            np.arange(1.5, 3.0, 0.01),
        )
    )
    prior_volume_zid = np.linspace(0, len(zgrid) - 1, 10).astype(int)

    lines = {
        "OII": 3726.8,
        "OIII_1": 4959.0,
        "OIII_2": 5007.0,
        "Halpha": 6562.8,
        "Hbeta": 4861,
        "Lyalpha": 1215.7,
        "NII_1": 6548.0,
        "NII_2": 6583.0,
        "SII_1": 6716.44,
        "SII_2": 6730.82,
    }

    # Median values from Alarcon2020
    line_ratios = {
        "OII": 1.0,
        "OIII_1": 0.31892333 * (1.0 / 3.0),
        "OIII_2": 0.31892333,
        "Halpha": 0.8227787,
        "Hbeta": 0.27377869,
        "Lyalpha": 2.0,
        "NII_1": 0.8227787 * 0.35 * 0.3,
        "NII_2": 0.8227787 * 0.35,
        "SII_1": 0.8227787 * 0.35,
        "SII_2": 0.8227787 * 0.35,
    }

    noext = [
        ["Ell1_A_0", "Ell4_A_0"],
        ["Ell4_A_0", "Ell7_A_0"],
        ["Ell7_A_0", "Sc_A_0"],
        ["Sc_A_0", "SB0_A_0"],
        ["SB0_A_0", "SB4_A_0"],
        ["SB4_A_0", "SB8_A_0"],
        ["SB8_A_0", "SB11_A_0"],
    ]

    calzetti = [
        ["Sc_A_0", "SB0_A_0"],
        ["SB0_A_0", "SB4_A_0"],
        ["SB4_A_0", "SB8_A_0"],
        ["SB8_A_0", "SB11_A_0"],
    ]

    EBV_values = np.arange(0.05, 0.501, 0.05)
    df = pd.DataFrame()
    ext_law = "none"
    for seds in noext:
        S = pd.Series({"ext_law": ext_law, "EBV": 0.0})
        S["seds"] = seds
        S["lines"] = lines
        S["line_ratios"] = line_ratios
        S["zgrid"] = zgrid
        S["nb2bb"] = nb2bb
        S["norm_band"] = norm_band
        S["ref_mag"] = ref_mag
        S["prior_volume_zid"] = prior_volume_zid
        df = df.append(S, ignore_index=True)

    ext_law = "SB_calzetti_bump2"
    for seds in calzetti:
        for EBV in EBV_values:
            S = pd.Series({"ext_law": ext_law, "EBV": EBV})
            S["seds"] = seds
            S["lines"] = lines
            S["line_ratios"] = line_ratios
            S["zgrid"] = zgrid
            S["nb2bb"] = nb2bb
            S["norm_band"] = norm_band
            S["ref_mag"] = ref_mag
            S["prior_volume_zid"] = prior_volume_zid
            df = df.append(S, ignore_index=True)
    # This applies to all configurations.

    test_data_dir = "/Users/alarcon/Documents/source/bcnz/test/data/"
    filter_dir = test_data_dir + "filt_70bands_noirach/"
    sed_dir = test_data_dir + "seds/"
    ext_dir = test_data_dir + "ext_laws/"
    filter_dir_calib = test_data_dir + "calib_filters/"
    df["sed_dir"] = sed_dir
    df["filter_dir"] = filter_dir
    df["filter_dir_calib"] = filter_dir_calib
    df["ext_dir"] = ext_dir


    return df
