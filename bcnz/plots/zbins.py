#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import numpy as np
from matplotlib import pyplot as plt

def zbins(cat, bins=[100, 125, 143, 144, 150, 175]):
    """Plot redshift distributions when binning on photo-z."""

    zbins = np.arange(0., 1.5, 0.005)

    fig, A = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches([2*6, 2*4])


    for i,bin_nr in enumerate(bins):
        ax = A.flatten()[i]
        
        sub = cat[(zbins[bin_nr] <= cat.zb) & (cat.zb < zbins[bin_nr+1])]
        sub.zs.hist(ax=ax, bins=50, histtype='step', color='k', lw=2.)
        
        label = '[{0}, {1:.5})'.format(zbins[bin_nr], zbins[bin_nr+1])
        ax.axvline(sub.zb.median(), lw=1.5, color='r', label='zp')

        ax.set_title(label)
        
        ax.set_xlim(0, 1.5)
        ax.legend()
        
    for ax in A[-1]:
        ax.set_xlabel('$z_{\\rm s}$', size=14)
