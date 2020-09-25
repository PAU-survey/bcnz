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
#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import pandas as pd

# These are not configurable right now, but this should not
# be too difficult to implement.
config = {
    'OII': 1.0,
    'OIII_1': 0.25*0.36,
    'OIII_2': 0.75*0.36,
    'Hbeta': 0.61,
    'Halpha': 1.77,
    'Lyalpha':  2.,
    'NII_1': 0.3 * 0.35 * 1.77,  # Paper gave lines relative to Halpha.
    'NII_2': 0.35 * 1.77,
    'SII_1': 0.35,
    'SII_2': 0.35
}

# The line locations are considered fixed.
line_lmb = {
    'OII': 3726.8,
    'OIII_1': 4959.,
    'OIII_2': 5007.,
    'Halpha': 6562.8,
    'Hbeta': 4861,
    'Lyalpha': 1215.7,
    'NII_1': 6548.,
    'NII_2': 6583.,
    'SII_1': 6716.44,
    'SII_2': 6730.82
}


def line_ratios():
    """Emission line ratios."""

    lmb = pd.Series(line_lmb)
    ratios = pd.Series(config)
    df = pd.DataFrame({'lmb': lmb, 'ratio': ratios})

    return df
