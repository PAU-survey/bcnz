# encoding: UTF8
from __future__ import print_function

import pdb

import bcnz

class pzcat:
    def __init__(self, conf, zdata, obs_file, out_name):

        self.conf = conf
        self.zdata = zdata
        self.obs_file = obs_file
        self.out_name = out_name

        self.conf['nmax'] = 10000

    def estimate_photoz(self, out_file):
        """Estimate the photoz for one input file."""

        nmax = self.conf['nmax']
        cols_keys, cols = bcnz.lib.bcnz_flux.get_cols(self.conf, self.zdata) 
        filters = self.zdata['filters']

        read_cat = bcnz.io.ascii.read_cat
        write_cat = bcnz.io.ascii.write_cat

        cat_in = read_cat(self.obs_file, nmax, cols_keys, cols, filters)
        f_out = write_cat(self.conf, out_file)

        for data in cat_in:
            data = bcnz_flux.post_pros(self.conf, data)
            f_obs, ef_obs = bcnz_flux.fix_fluxes(self.conf, self.zdata, data) 

            ids = data['ids']
            m_0 = data['m_0']
            z_s = data['z_s']
  
            f_obs, ef_obs = bcnz_norm.norm_data(self.conf, self.zdata, f_obs, ef_obs)

            inst = bcnz_chi2.chi2_inst(self.conf, self.zdata, f_obs, ef_obs, m_0, \
                          z_s, ids, 100)


            ng = len(f_obs)
            for block in inst.blocks():
                pz_table.append(block)

        f_out.close()
