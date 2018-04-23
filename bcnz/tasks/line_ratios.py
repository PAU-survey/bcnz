#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import pandas as pd

class line_ratios:
    """Task for configuring the emission line ratios."""

    version = 1.0

    config = {
      'OII': 1.0,
      'OIII_1': 0.25*0.36,
      'OIII_2': 0.75*0.36,
      'Hbeta': 0.61,
      'Halpha': 1.77,
      'Lyalpha':  2.,
      'NII_1': 0.3 * 0.35 * 1.77, # Paper gave lines relative to Halpha.
      'NII_2': 0.35 * 1.77,
      'SII_1': 0.35,
      'SII_2': 0.35
    }

    def run(self):
        self.output.result = pd.Series(self.config)
