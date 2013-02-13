#!/usr/bin/env python
# encoding: UTF8

import bcnz

class pzcat:
    def __init__(self, myconf):
        self.conf = bcnz.libconf(myconf)

    def run(self):
        # Estimate the photoz
        zdata = bcnz.zdata.zdata(self.conf)
        model = bcnz.model.model(self.conf, zdata)

        # Bad hack, the interface needs to be improved.
        import bcnz_main
        bcnz_main.pzcat_local(self.conf, zdata, model)
