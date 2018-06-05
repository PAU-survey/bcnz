#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb

descr = {}

class apply_zp:
    """Apply zero-points per band."""

    version = 1.02
    config = {}

    # Note: One can consider adding the functionality
    # of specifying some zero-points by the configuration.
    def entry(self, galcat, zp):
        # Applying this inline is simpler.
        for band, zp_val in zp.items():
            galcat[('flux', band)] *= zp_val
            galcat[('flux_err', band)] *= zp_val

        return galcat

    def run(self):
        galcat = self.input.galcat.result
        zp = self.input.zp.result

        self.output.result = self.entry(galcat, zp)
