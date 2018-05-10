#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import xarray as xr

descr = {'chi2_max': 'Limit on chi2 values (0 no cut)'}

class zp_pbadger:
    """A new type of zero-points."""

    version = 1.0
    config = {'chi2_max': 50}

    # Having this as a separate task might be a bit overkill. Potentially
    # one can select on galaxy properties (e.g SN) here before making the
    # zero-points.
    def entry(self, chi2, ratio):
        # Combine these into a dataset.
        chi2 = chi2.to_xarray().chi2
        ratio = ratio.to_xarray().ratio
        comb = xr.Dataset({'chi2': chi2, 'ratio': ratio})

        if self.config['chi2_max']:
            minval = comb.chi2.min(dim='chunk')
            comb = comb.sel(gal=minval < self.config['chi2_max'])

        best_chunk = comb.chi2.argmin(dim='chunk')
        best_ratio = comb.ratio.isel_points(chunk=best_chunk, gal=\
                     range(len(best_chunk)))

        zp = best_ratio.median(dim='points')
        zp = zp.to_series()
 
        # For some reason I divided when applying the zero-points.. 
        zp = 1./zp

        return zp

    def run(self):
        store = self.input.galfit.get_store()
        chi2 = store['chi2']
        ratio = store['ratio']
        store.close()

        self.output.result = self.entry(chi2, ratio)
