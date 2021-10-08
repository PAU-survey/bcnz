#!/usr/bin/env python

#import ipdb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def get68(sub):
    """The sigma68 value."""

    sig1 = 0.6827
    xa = 0.5 - sig1 / 2.
    xb = .5 + sig1 / 2.

    quant = sub.dx.quantile(q=[xa,xb])
    sig68 = 0.5*(quant[xb] - quant[xa])

    return sig68

def nmad(sub):
    """NMAD definition."""

    delta_z = sub.zb - sub.zs
    X = 1.48*(np.abs(delta_z - delta_z.median() / (1. + sub.zs))).median()

    return X

def sig68(cat, cut_key):
    """Estimate sigma_68 for different magnitude cuts."""

    S = pd.Series()
    for q in [1.0, 0.8, 0.5, 0.2]:
        sub = cat[cat[cut_key] < cat[cut_key].quantile(q)]
        S[q] = get68(sub)

    return S

def outl(cat):
    """Outlier fraction in catalogue."""

    return (0.02 < cat.dx.abs()).mean()

def bias(cat):
    """Bias in the catalogue."""

    return (cat.zb - cat.zs).median()


def stats(cat):
    """Estimate basic statistics for catalogue."""

    S = pd.Series()
    S['median'] = cat.dx.median()

    return S 

def _core_bins(sub, cut_key):
    df = pd.DataFrame()
    for q in [0.2, 0.3, 0.4, 0.5, .8, 1.0]:
        sub2 = sub[sub[cut_key] <= sub[cut_key].quantile(q)]
        S = pd.Series({'q': q, 'sig68': get68(sub2),
                       'nmad': nmad(sub2), 
                       'outl': outl(sub2),
                       'bias': bias(sub2)})

        df = df.append(S, ignore_index=True)

    return df 

def cum_bins(cat, bins, quantity, cut_key):
    """Estimates the quantities in cumulative bins."""

    df = pd.DataFrame()
    for lim_cut in bins:
        sub = cat[cat[quantity] < lim_cut]

        part = _core_bins(sub, cut_key)
        part['xbin_val'] = lim_cut
        df = df.append(part, ignore_index=True)

    return df

def normal_bins(cat, bins, quantity, cut_key='Qz'):
    df = pd.DataFrame()
    for key,sub in cat.groupby(pd.cut(cat[quantity], bins)):
        part = _core_bins(sub, cut_key)

        part['xbin_val'] = sub[quantity].mean()
        df = df.append(part, ignore_index=True)

    return df


def plot1(L, dz=0.001):
    """Histogram of the photo-z values."""

    bins = np.arange(-0.5, 0.5, dz)
    K = {'histtype': 'step', 'normed': True, 'bins': bins}

    for lbl, cat in L:
        cat.dx.hist(label=lbl, **K)
    
    plt.legend()
    plt.show()

def _add_key(cat, cut_key):
    if 'qual' in cat.columns:
        return

    if cut_key in ['odds']:
        cat['qual'] = cat[cut_key]
    elif cut_key in ['pz_width', 'Qz', 'qz']:
        cat['qual'] = 1./cat[cut_key]
    else:
        print(cat)
        raise ValueError('Unknown error: {}'.format(cut_key))

#
# Single sigma68 plot. Should be fixed up!
#
#def sigma68(L, cut_key='qz', ls=[':', '--', '-', ':'], q=[1.0, 0.8,0.5,0.2], color=['g', 'k','r','b'],
#          yval='sig68', ax=None):
#    """Sigma-68 as a function of magnitude."""
#
#    if not isinstance(L, list):
#        L = [('PAU', L)]
#
#    if isinstance(ax, type(None)):
#        ax = plt.gca()
#
#
#    if not isinstance(cut_key, list):
#        cut_key = len(L)*[cut_key]
#
#    K = np.linspace(19., 22.55)
#    for i,(lbl1,cat) in enumerate(L):
#        _add_key(cat, cut_key[i])
#
#        df_new = cum_bins(cat, K, cut_key[i])
#
#        for col,qi in zip(color, q):
#            sub_new = df_new[df_new.q == qi]
#
#            lbl = '{}, {}%'.format(lbl1, 100*qi)
#            ax.plot(sub_new.mag, sub_new[yval], color=col, lw=1.2, ls=ls[i], label=lbl)
#
#    ax.axhline(0.0035, ls='--', color='k', alpha=0.5)
#    ax.set_xlabel('$\mathrm{i_{AB} < i_{Auto}}$', size=14)
#
#    
#    ylabelD = {'sig68': '$\sigma_{68}\, /\, (1+z)$', 
#               'nmad': 'NMAD'} 
#
#    ax.set_ylabel(ylabelD[yval], size=12)
#
#    ax.set_yscale('log')
#    ax.grid(which='both')
#
#    ax.legend(prop={'size': 8})
#

def metrics(L, cut_key='qz', ls=[':', '--', '-', ':'], q=[1.0, 0.8,0.5,0.2], color=['g', 'k','r','b']):
    """Sigma-68, outlier rate and  as a function of magnitude, photo-z and spec-z."""

    if not isinstance(L, list):
        L = [('PAU', L)]

    if not isinstance(cut_key, list):
        cut_key = len(L)*[cut_key]

    # Nested..
    def _plot_panel(ax, bin_function, edges, metric, xquantity):
        for i,(lbl1,cat) in enumerate(L):
            _add_key(cat, cut_key[i])

            #df_new = cum_bins(cat, edges, cut_key[i])
            df_new = bin_function(cat, edges, xquantity, cut_key[i])

            for col,qi in zip(color, q):
                sub_new = df_new[df_new.q == qi]

                lbl = '{}, {}%'.format(lbl1, 100*qi)
                ax.plot(sub_new.xbin_val, sub_new[metric], color=col, lw=1.2, ls=ls[i], label=lbl)

    # Magnitude cut..
    print('Update', 5)

    KD = {'I_auto': np.linspace(19., 22.55, 15),
          'zb': np.arange(0.1, 1.5, 0.1),
          'zs': np.arange(0.1, 1.5, 0.1)}


    # Here one can be more fancy. However, having a static configuration makes the code much
    # easier to read.
    fig, A = plt.subplots(nrows = 3, ncols = 3, sharex='col')

    k = 2.5
    fig.set_size_inches([k*6, k*4])

    for i,metric in enumerate(['sig68', 'outl', 'bias']):
        for j,xquantity in enumerate(['I_auto', 'zb', 'zs']):
            K = KD[xquantity]
            _plot_panel(A[i,j], normal_bins, K, metric, xquantity)


    size = 16
    A[0,0].set_ylabel('$\sigma_{68}\, /\, (1+z)$', size=size)
    A[1,0].set_ylabel('Outlier fraction', size=size)
    A[2,0].set_ylabel('Bias', size=size)

    for ax in A[0]:
        ax.axhline(0.0035, ls='--', color='k', alpha=0.5)
        ax.set_yscale('log')

    for ax in A[2]:
        ax.set_ylim((-0.01, 0.01))

    for ax in A.flatten():
        ax.grid(which='both')

    A[-1,0].set_xlabel('$\mathrm{i_{AB} < i_{Auto}}$', size=size)
    A[-1,1].set_xlabel('$z_{\\rm b}$', size=size)
    A[-1,2].set_xlabel('$z_{\\rm spec}$', size=size)

    A[0,0].legend(prop={'size': 8})
