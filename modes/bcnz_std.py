# encoding: UTF8
from __future__ import print_function
import pdb
import numpy as np

import bpz_flux
import bpz_output

import loadparts
import bcnz_chi2
import bcnz_flux
import bcnz_norm
import bcnz_output

class standard:
    def __init__(self, conf, zdata, obs_file, out_name, mstep, ninterp):

        # You might wonder why.. This is only a transition..
        self.conf = conf
        self.zdata = zdata
        self.obs_file = obs_file
        self.out_name = out_name

        self.ninterp = ninterp
        self.mstep = mstep
        self.conf['nmax'] = 10000

    def run_file(self): #, obs_file):
        obs_file = self.obs_file

        # Prepare output file.
#self.col_pars)
        out_format, hdr, has_mags = bpz_output.find_something(self.conf, self.zdata)

        output = bcnz_output.output_file(self.out_name)

        if self.conf['get_z']:
            bpz_output.add_header(self.conf, output, self.out_name, hdr)


        nmax = self.conf['nmax']
        cols_keys, cols = bpz_flux.get_cols(self.conf, self.zdata) 
        tmp = loadparts.loadparts(obs_file, nmax, cols_keys, cols)

        for data in tmp:
            data = bpz_flux.post_pros(data, self.conf)
            ids,f_obs,ef_obs,m_0,z_s = bcnz_flux.fix_fluxes(self.conf, self.zdata, data) 

            f_obs, ef_obs = bcnz_norm.norm_data(self.conf, self.zdata, f_obs, ef_obs)

            inst = bcnz_chi2.chi2(self.conf, self.zdata, f_obs, ef_obs, m_0, self.mstep, self.ninterp, \
                          z_s, 100)


            ng = len(f_obs)

            for ig in range(ng):
                iz_ml, t_ml, red_chi2, pb, p_bayes, iz_b, zb, o, \
                it_b, tt_b, tt_ml,z1,z2,opt_type = inst(ig)

                # <standard BPZ>
                salida=[ids[ig],zb,z1,z2,tt_b+1,o,self.zdata['z'][iz_ml],tt_ml+1,red_chi2]
                if 'Z_S' in self.zdata['col_pars']: salida.append(z_s[ig])
                if has_mags: salida.append(m_0[ig]-self.conf['delta_m_0'])
                if 'OTHER' in self.zdata['col_pars']: salida.append(other[ig])

                output.write(out_format % tuple(salida)+'\n')

                # </standard BPZ>
