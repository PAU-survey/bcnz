# encoding: UTF8
from __future__ import print_function

import pdb

import bcnz

class pzcat:
    def __init__(self, myconf, in_iter, out_table):
        self.conf = bcnz.libconf(myconf)
        zdata = bcnz.zdata.zdata(self.conf)
        self.zdata = bcnz.model.add_model(self.conf, zdata)

        self.in_iter = in_iter
        self.out_table = out_table

    def run(self):
        """Estimate the photoz for one input file."""

        in_iter = self.in_iter
        out_table = self.out_table

        for data in in_iter:
            data = bcnz.observ.post_pros(self.conf, self.zdata, data)
            chi2 = bcnz.fit.chi2(self.conf, self.zdata, data)

            for block in inst.blocks():
                out_table.append(block)

        out_table.close()
