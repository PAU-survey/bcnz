def test_offset():
    import loadparts
    import libpz
    nmax = 50000
    nmax = 10000
    cols_keys, cols = bpz_flux.get_cols(conf, col_pars, flux_cols, eflux_cols)
    tmp = loadparts.loadparts(obs_file, nmax, cols_keys, cols)

    tmp = list(tmp)
    data = tmp[0]
    data = bpz_flux.post_pros(data, conf)
#    pdb.set_trace()
    t1 = time.time()
    def find_zp(X_offsets):
        ids,f_obs,ef_obs,m_0,z_s = bpz_flux.mega_function(conf,data,undet,unobs,
                                   zp_errors, X_offsets,filters,cals,col_pars)


        ng = len(f_obs)
    #    ids,f_obs,ef_obs,m_0,z_s = bpz_flux.mega_function(conf,data,undet,unobs,
    #                               zp_errors,zp_offsets,filters,cals,col_pars)

        inst = bcnz.C(f_obs, ef_obs, f_mod, z, m_0, mstep, ninterp, \
                      P_MIN, ODDS, MIN_RMS, z_s, 300)

        z_p = np.zeros((ng))
        odds = np.zeros((ng))
        for ig in range(ng):
            iz_ml, t_ml, red_chi2, pb, p_bayes, iz_b, zb, o, it_b, tt_b, tt_ml,z1,z2,opt_type = inst(ig)
            z_p[ig] = z[iz_ml]
            odds[ig] = o

        return z_p, odds

    def weight(z_p, z_s,odds):
        rms = np.sum((z_p-z_s)**2) / len(z_p)

        cat = np.vstack([z_s, z_p, odds]).T
        sigma = libpz.sigma(cat, np.array([0.,2.]))

        #pdb.set_trace()
#        return rms
        return sigma

    def f(X):
        t1 = time.time()
        z_p, odds = find_zp(X)
        z_s = data['z_s']

        w = weight(z_p, z_s, odds)
        print('w', X, w)

        t2 = time.time()
        return w

    import scipy.optimize
    from scipy.optimize import fmin_bfgs
    x0 = zp_offsets.copy()
    F = scipy.optimize.brute

    f(np.zeros(5))
    pre_opt = np.array([0.2,-0.1,-0.2,-0.3,0.])
    ans = fmin_bfgs(f, pre_opt)
#    F(f, 5*[[-.3,.3]], Ns=7, finish=fmin_bfgs)

#    scipy.optimize.anneal(f, x0,lower=-.5, upper=.5)
    pdb.set_trace()
#    a1 = np.array([0.,0.,0.,0.,0.2])
#    a1 = np.array([0.,0.,0.,0.,0.5])
    print(f(zp_offsets))
    print(a1, f(a1))
    #print(f(zp_offsets))

#    ans = find_zp(zp_offsets)

    t2 = time.time()
    pdb.set_trace()

    def weight(x,y):
        pdb.set_trace()
