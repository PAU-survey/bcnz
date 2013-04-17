# encoding: UTF8
from __future__ import print_function

import pdb

import bcnz

class pzcat:
    """Photoz catalog estimation.
       Run the photo-z for one sample.
    """

    def __init__(self, myconf):
        self.conf = bcnz.libconf(myconf)

    # Fields to properly implement.
    config_schema = """{}"""
    input_schema = """{}"""
    output_schema = """{}"""
    config_sample = """{}"""
    input_sample =  """{}"""
    output_sample = """{}"""

    @classmethod
    def check_config(cls, inp): pass

    @classmethod
    def check_input(cls, inp): pass

    @classmethod
    def check_output(cls, out): pass

    @property
    def config_schema(self):
        return self.conf.schema()
#        pdb.set_trace()

    @property
    def config_sample(self):
        return self.conf.config_sample()

    def _run_iter(self, iter_in, out_table):
        self.in_iter.open()
        self.out_table.open()

        if not zdata: 
            zdata = bcnz.zdata.zdata(self.conf)
            self.zdata = bcnz.model.add_model(self.conf, zdata)
        else:
            self.zdata = zdata

        for data in self.in_iter:
            data = bcnz.observ.post_pros(self.conf, self.zdata, data)
            fit = bcnz.fit.chi2(self.conf, self.zdata, data)
            for block in fit.blocks():
                self.out_table.append(block)

        self.in_iter.close()
        self.out_table.close()

    def run(self):
        """Estimate the photoz for one input file."""

        raise NotImplementedError()
