#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import numpy as np
import pandas as pd

descr = {'ngal': 'Number of galaxies to select'}

class limit_ngal:
    """Limit the number of galaxies."""

    # One could in theory apply the narrow band selection
    # a second time. The problem is one need to be very 
    # careful that these are actually the same.
    version = 1.0
    config = {'ngal': 0}

    def entry(self, cat_in):
        ngal = self.config['ngal']
        if ngal:
            cat = cat_in.sample(ngal)
        else:
            cat = cat_in

        return cat

    def run(self):
        cat_in = self.input.galcat.result
        self.output.result = self.entry(cat_in)
