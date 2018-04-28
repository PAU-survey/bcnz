#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import pandas as pd

class line_ratios:
    """Task for configuring the emission line ratios."""

    version = 1.01

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

    # The line locations are considered fixed.
    line_loc = {
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

    def entry(self):
        loc = pd.Series(self.line_loc)
        ratios = pd.Series(self.config)
        df = pd.DataFrame({'loc': loc, 'ratio': ratios})

        return df

    def run(self):
        self.output.result = self.entry()
