#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb

class chi2_add:
    """Adds together chi2 arrays."""

    version = 1.0
    config = {}

    def load_chi2(self):
        """Load the chi2 arrays."""
        D = {}
        for key, dep in self.input.depend.items():
            if not key.startswith('chi2_'):
                continue

            new_key = key.replace('chi2_','')
            D[new_key] = dep.result

        return D

    def entry(self):
        D = self.load_chi2()
        comb_chi2 = sum(D.values())

        return comb_chi2

    def run(self):
        self.output.result = self.entry()

