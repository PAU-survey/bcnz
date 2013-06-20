#!/usr/bin/env python
# encoding: UTF8

import copy
import os
import pdb
import sys

import bcnz

class conf(dict,object):
    msg_z = 'zmax <= zmin is not allowed.'
    msg_overwrite = 'Overwriting some input files.'

    def __init__(self, myconf):
        self.update(bcnz.config.conf['standard'].copy()) # HACK

        if 'c' in myconf:
            extra_conf = myconf['c']
            assert extra_conf in bcnz.config.conf, 'No configuration for: {}'.format(extra_conf)

            self.update(bcnz.config.conf[extra_conf].copy())

        self.update(myconf)
        self._test_zrange()

    def config_sample(self):
        def flatten(d, pre=[]):
            """Flatten a dictionary tree with parameters."""

            if isinstance(d, dict):
                res = {}
                for key,val in d.iteritems():
                    res.update(flatten(val,pre+[key]))

                return res
            else:
                return {'.'.join(pre): d}

        return flatten(self)

    def schema(self):
        """JSON schema Tallada-style."""

        global tojson
        tojson = {str: 'string', int: 'integer', float: 'float',
                  list: 'list', bool: 'bolean', unicode: 'str'}

        flatten_conf = self.config_sample()

        defconf = copy.deepcopy(self)
        injson = ["'" + key + "': { 'type': '"+tojson[type(val)]+"'}" for \
                  key,val in flatten_conf.iteritems()]

        # Produce a string with the schema.
        txt = """{'title': 'BCNZ Schema', 'type': 'object', 'properties': {"""
        txt += ',\n'.join(injson)+'}'

        return txt

    def _test_zrange(self):

        assert self['zmin'] < self['zmax'], self.msg_z
