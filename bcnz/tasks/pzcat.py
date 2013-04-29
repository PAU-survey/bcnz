# encoding: UTF8
from __future__ import print_function

import pdb

import bcnz
import bcnz.fit
import bcnz.observ

class pzcat(object):
    """Photoz catalog estimation.
       Run the photo-z for one sample.
    """

    def __init__(self, myconf):
        self.config = bcnz.libconf(myconf)

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
        return self.config.schema()

    @property
    def config_sample(self):
        return self.config.config_sample()

    def _run_iter(self, in_iter, out_table):
        in_iter.open()
        out_table.open()
        
        zdata = bcnz.zdata.zdata(self.config)
        zdata = bcnz.model.add_model(self.config, zdata)
        
        for data in in_iter:
            data = bcnz.observ.post_pros(self.config, zdata, data)
            fit = bcnz.fit.chi2(self.config, zdata, data)
            for block in fit.blocks():
                out_table.append(block)
        
        in_iter.close()
        out_table.close()

    def run(self):
        """Estimate the photoz for one input file."""

        raise NotImplementedError()
