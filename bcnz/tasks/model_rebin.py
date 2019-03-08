#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import time
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev, splint

descr = {
  'zmin': 'Minimum redshift',
  'zmax': 'Maximum redshift',
  'dz': 'Grid width in redshift'
}

class model_rebin:
    """Rebing the redshift grid of the model."""

    version = 1.02
    config = {'zmin': 0.01, 'zmax': 1.2, 'dz': 0.001}
  
    # This rebinning is probably not needed. Instead on can estimate
    # the model in the correct binning in the first place. I need this
    # by now to compare the pipelines. 

    def entry(self, model):
        C = self.config
        zgrid = np.arange(C['zmin'], C['zmax']+C['dz'], C['dz'])

        inds = ['band', 'sed', 'ext_law', 'EBV']
        model = model.reset_index().set_index(inds)

        print('starting to rebin')
        t1 = time.time()
        rebinned = pd.DataFrame()
        for key in model.index.unique():
            sub = model.loc[key]
            spl = splrep(sub.z, sub.flux)

            part = pd.DataFrame({'z': zgrid, 'flux': splev(zgrid, spl, ext=2)})

            # I needed to set these manually...
            for k1, v1 in zip(model.index.names, key):
                part[k1] = v1

            rebinned = rebinned.append(part)

        print('time', time.time() - t1)
        rebinned = rebinned.reset_index().set_index(inds+['z'])

        return rebinned

    def run(self):
        model = self.input.model.result
        self.output.result = self.entry(model)

