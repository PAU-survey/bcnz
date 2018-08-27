#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb

descr = {'norm_bb': 'If separately normalizing the broad bands',
         'norm_filter': 'Band to normalize to'}

class apply_zp:
    """Apply zero-points per band."""

    version = 1.02
    config = {'norm_bb': True,
              'norm_filter': 'subaru_r'}

    def normalize_zp(self, zp):
        """One need to normalize the zero-points for the broad bands
           when the calibration has been run with a free amplitude.
        """

        print('In normalize..')
        BBlist = list(filter(lambda x: not x.startswith('NB'), zp.index))
        norm_val = zp.loc[self.config['norm_filter']]

        for band in BBlist:
            zp.loc[band] /= norm_val


    def entry(self, galcat, zp):
        if self.config['norm_bb']:
            self.normalize_zp(zp)

        # Applying this inline is simpler.
        for band, zp_val in zp.items():
            galcat[('flux', band)] *= zp_val
            galcat[('flux_err', band)] *= zp_val

        return galcat

    def run(self):
        galcat = self.input.galcat.result
        zp = self.input.zp.result

        self.output.result = self.entry(galcat, zp)
