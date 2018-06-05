#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import numpy as np
import pandas as pd
import xarray as xr
import inter_calib

descr = {'fit_bands': 'Which bands to use in the fit',
         'SN_min': 'Minimum SN',
         'cosmos_scale': 'If changing the flux scale'}

class find_em(inter_calib.inter_calib):
    """Fine the emission line ratios."""

    # This code reuse some of the parts used for the intercalibration.
    # One should not actually need to also code the photo-z minimization
    # here, but I have not updated the inter_calib version yet....
    version = 1.012
    config = {'fit_bands': [],
              'Niter': 1000,
              'SN_min': -20,
              'cosmos_scale': True,
              'free_ampl': False,
              'Nskip': 10}

    def chi2_min_free(self, f_mod, flux, flux_err):
        """Minimize the chi2 expression."""

        # This method is adapted to separately output the continuum and the
        # emission line modelling!

        var_inv = 1./flux_err**2

        NBlist = list(filter(lambda x: x.startswith('NB'), flux.band.values))
        BBlist = list(filter(lambda x: not x.startswith('NB'), flux.band.values))

        A_NB = np.einsum('gf,gfs,gft->gst', var_inv.sel(band=NBlist), \
               f_mod.sel(band=NBlist), f_mod.sel(band=NBlist))
        b_NB = np.einsum('gf,gf,gfs->gs', var_inv.sel(band=NBlist), flux.sel(band=NBlist), \
               f_mod.sel(band=NBlist))
        A_BB = np.einsum('gf,gfs,gft->gst', var_inv.sel(band=BBlist), \
               f_mod.sel(band=BBlist), f_mod.sel(band=BBlist))
        b_BB = np.einsum('gf,gf,gfs->gs', var_inv.sel(band=BBlist), flux.sel(band=BBlist), \
               f_mod.sel(band=BBlist))

        # Testing to scale to the narrow bands. In that case the code above is not needed.
        scale_to = NBlist
        var_inv_NB = var_inv.sel(band=scale_to)
        flux_NB = flux.sel(band=scale_to)
        f_mod_NB = f_mod.sel(band=scale_to)
        S1 = (var_inv_NB*flux_NB).sum(dim='band')

        # Since we need these entries in the beginning...
        k = np.ones(len(flux))
        b = b_BB + k[:,np.newaxis]*b_NB
        A = A_BB + k[:,np.newaxis,np.newaxis]**2*A_NB

        v = 100*np.ones_like(b)

        gal_id = flux.ref_id.values
        coords = {'ref_id': gal_id, 'band': f_mod.band}
        coords_norm = {'ref_id': gal_id, 'model': f_mod.sed}

        for i in range(self.config['Niter']):
            a = np.einsum('gst,gt->gs', A, v)

            m0 = b / a
            vn = m0*v

            v = vn
            # Extra step for the amplitude
            if 0 < i and i % self.config['Nskip'] == 0:
                # Testing a new form for scaling the amplitude...
                S2 = np.einsum('gf,gfs,gs->g', var_inv_NB, f_mod_NB, v)
                k = (S1.values/S2.T).T

                # Just to avoid crazy values ...
                k = np.clip(k, 0.1, 10)

                b = b_BB + k[:,np.newaxis]*b_NB
                A = A_BB + k[:,np.newaxis,np.newaxis]**2*A_NB


        # Yes, this is getting slightly complicated...
        isline = pd.Series(f_mod.sed).isin(['lines', 'OIII']).values
        isNB = np.array(list(map(lambda x: x.startswith('NB'), flux.band.values)))

        # The continuum modelling.
        L = []
        L.append(np.einsum('g,gfs,gs->gf', k, f_mod[:,isNB,~isline], v[:,~isline]))
        L.append(np.einsum('gfs,gs->gf', f_mod[:,~isNB,~isline], v[:,~isline]))
        Fcont = np.hstack(L)
        coords['band'] = NBlist + BBlist
        Fcont = xr.DataArray(Fcont, coords=coords, dims=('ref_id', 'band'))

        # Otherwise we get problems in the cases when emission lines are not added...
        if isline.any(): 
            L = []
            L.append(np.einsum('g,gfs,gs->gf', k, f_mod[:,isNB,isline], v[:,isline]))
            L.append(np.einsum('gfs,gs->gf', f_mod[:,~isNB,isline], v[:,isline]))
            Flines = np.hstack(L)
            coords['band'] = NBlist + BBlist
            Flines = xr.DataArray(Flines, coords=coords, dims=('ref_id', 'band'))
        else:
            Flines = Fcont.copy()
            Flines[:,:] = 0.

        F = Fcont + Flines
        chi2 = (var_inv*(F - flux)**2)

#        ipdb.set_trace()

        return chi2, Fcont, Flines


    def find_best_model(self, modelD, cont_model, lines_model, flux, flux_err, chi2):
        """Find the best flux model."""

        # Just get a normal list of the models.
        model_parts = [str(x.values) for x in cont_model.part]

        fmin = self.chi2_min_free
        for j,key in enumerate(model_parts):
            print('Part', j)

            chi2_part, Fcont, Flines = fmin(modelD[key], flux, flux_err)
            chi2[j,:] = chi2_part.sum(dim='band')

            cont_model[j,:] = Fcont
            lines_model[j,:] = Flines

        # Ok, this is not the only possible assumption!
        dims = ('ref_id', 'band')
        coords = {'ref_id': flux.ref_id, 'band': flux.band}
        best_part = chi2.argmin(dim='part')

        best_cont = cont_model.isel_points(ref_id=range(len(flux)), part=best_part)
        best_cont = xr.DataArray(best_cont, dims=dims, coords=coords)

        best_lines = lines_model.isel_points(ref_id=range(len(flux)), part=best_part)
        best_lines = xr.DataArray(best_lines, dims=dims, coords=coords)

        best_flux = xr.Dataset({'cont': best_cont, 'lines': best_lines})

        #F = best_flux.cont + best_flux.lines
        #ipdb.set_trace()

        return best_flux

    def entry(self, galcat):
        modelD = self.get_model(galcat.zs)
        flux, flux_err, chi2, zp_tot, cont_model = self._prepare_input(modelD, galcat)
        lines_model = cont_model.copy()

        best_flux = self.find_best_model(modelD, cont_model, lines_model, flux, flux_err, chi2)

        if self.config['cosmos_scale']:
            ab_factor = 10**(0.4*26)
            cosmos_scale = ab_factor * 10**(0.4*48.6)
            best_flux *= cosmos_scale

        return best_flux


    def run(self):
        galcat = self.input.galcat.result
        galcat = galcat[~np.isnan(galcat.zs)]
        best_flux = self.entry(galcat)

        path = self.output.empty_file('default')
        best_flux.to_netcdf(path)
