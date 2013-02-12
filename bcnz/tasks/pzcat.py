#!/usr/bin/env python
# encoding: UTF8

import bcnz

class pzcat:
    def __init__(self, myconf):
        self.conf = bcnz.libconf(myconf)

    def run(self):
        # Estimate the photoz
        zdata = bcnz.zdata.zdata(conf)
        model = bcnz.model.model(conf, zdata)

        # Bad hack, the interface needs to be improved.
        import bcnz_main
        bcnz_main.pzcat(conf,zdata,model)
