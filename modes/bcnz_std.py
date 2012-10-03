# encoding: UTF8
from __future__ import print_function
import os
import pdb
import shutil
import numpy as np

import bpz_flux

import loadparts
import obj_hash
import bcnz_chi2
import bcnz_flux
import bcnz_norm
import bcnz_output

class standard:
    def __init__(self, conf, zdata, obs_file, out_name):

        self.conf = conf
        self.zdata = zdata
        self.obs_file = obs_file
        self.out_name = out_name

        self.conf['nmax'] = 10000

        obj_name = obj_hash.hash_structure(self.conf)

        root_dir = self.conf['root']
        self.cache_dir = os.path.join(root_dir, 'cache')
        self.obj_path = os.path.join(self.cache_dir, obj_name)


    def relink(self):
        """Link output file name to the object file."""

        if os.path.isfile(self.out_name):
            shutil.move(self.out_name, "%s.bak" % self.out_name)

        if os.path.islink(self.out_name):
            os.remove(self.out_name)

        os.symlink(self.obj_path, self.out_name)

    def run_file(self):
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        if os.path.isfile(self.obj_path):
            self.relink()
            return 

        self.estimate_photoz()
        self.relink()

    def estimate_photoz(self):
        """Estimate the photoz for one input file."""

        obs_file = self.obs_file
        out_file = bcnz_output.output_file(self.obj_path)

        if self.conf['get_z']:
            header = bcnz_output.create_header(self.conf, obs_file)
            header = ['{0}\n'.format(x) for x in header]
            out_file.writelines(header)

        nmax = self.conf['nmax']
        cols_keys, cols = bpz_flux.get_cols(self.conf, self.zdata) 
        tmp = loadparts.loadparts(obs_file, nmax, cols_keys, cols)

        ndesi = self.conf['ndesi']
        columns = self.conf['order']+self.conf['others']
        rest = [('{%s:.%sf}' % (x, ndesi)) for x in columns]
        out_format = '{id} ' + ' '.join(rest) + '\n'

        for data in tmp:
            data = bpz_flux.post_pros(data, self.conf)
            ids,f_obs,ef_obs,m_0,z_s = bcnz_flux.fix_fluxes(self.conf, self.zdata, data) 

            f_obs, ef_obs = bcnz_norm.norm_data(self.conf, self.zdata, f_obs, ef_obs)

            inst = bcnz_chi2.chi2(self.conf, self.zdata, f_obs, ef_obs, m_0, \
                          z_s, 100)


            ng = len(f_obs)

            for ig in range(ng):
                iz_ml, t_ml, red_chi2, pb, p_bayes, iz_b, zb, odds, \
                it_b, tt_b, tt_ml,z1,z2,opt_type = inst(ig)

                # Temporary before moving it into the chi2 estimation code.
                gal = {'id': ids[ig], 'zb': zb, 'zb_min':  z1, 'zb_max': z2, \
                       't_b': tt_b+1, 'odds': odds, 'z_ml': self.zdata['z'][iz_ml], \
                       't_ml': tt_ml+1, 'chi2': red_chi2, 'z_s': z_s[ig], \
                       'm_0': m_0[ig] - self.conf['delta_m_0']}

                out_file.write(out_format.format(**gal))

