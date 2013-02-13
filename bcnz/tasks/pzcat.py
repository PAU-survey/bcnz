# encoding: UTF8
from __future__ import print_function

import pdb

import bcnz

class pzcat:
    def __init__(self, myconf, in_iter, out_table):
        self.conf = bcnz.libconf(myconf)
        self.zdata = bcnz.zdata.zdata(self.conf)

        self.in_iter = in_iter
        self.out_table = out_table

    def run(self):
        """Estimate the photoz for one input file."""

        in_iter = self.in_iter
        out_table = self.out_table

        for data in in_iter:
            data = bcnz_flux.post_pros(self.conf, data)

            f_obs, ef_obs = bcnz_flux.fix_fluxes(self.conf, self.zdata, data) 
            chi2 = bcnz.chi2.chi2(self.conf, self.zdata, f_obs, ef_obs)

            for block in inst.blocks():
                out_table.append(block)

        out_table.close()
