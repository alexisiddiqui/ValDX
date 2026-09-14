#!/usr/bin/env python

# Plots L-curve of gamma vs MSD for the results of reweighting with individual residues

import cbf_colors
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import os

### Define defaults for matplotlib plots
plt.style.use('../styles+swatches/cbf_8color_cycler_paper_80mm.mplstyle')


colors = [ 'cbf_key', 'cbf_orange', 'cbf_bluish_green' ]

# Plot L-curve
names = [ '10', 'noise_10^-2', 'noise_10^-1' ]
folders = [ './data_folder/' + i for i in names ]
powers = [ -2, -1, 0, 1, 2, 3 ]

def make_filelist(folder, powers):
    files1 = []
    files2 = []
    files3 = []
    gammas = []
    for p in powers:
        for j in range(1,10):
            files1.append(os.path.join(folder[1], "mixed_gamma_%sx10^%s_work.dat" % (j,p)))
            files2.append(os.path.join(folder[0], "mixed_gamma_%sx10^%s_work.dat" % (j,p)))
            files3.append(os.path.join(folder[2], "mixed_gamma_%sx10^%s_work.dat" % (j,p)))
            gammas.append(j * 10**p)
    return files1, files2, files3, gammas
         
def files_to_array(fnames, gammas):
    """Read in data fom list of files with np.loadtxt.
       Returns array of shape (n_files, n_data_per_file)
       Empty files are dropped"""
    l = []
    foundgammas = []
    for f, g in zip(fnames, gammas):
        try:
            _ = np.loadtxt(f)
        except IOError:
            continue
        if _.shape == (0,):
            continue
        else:
            l.append(_)
            foundgammas.append(g)
    try:
        return np.stack(l, axis=0), foundgammas
    except ValueError:
        raise ValueError("Error in stacking files read with np.loadtxt - are they all the same length?")

allfiles = make_filelist(folders, powers)
datatup_missing = files_to_array(allfiles[0], allfiles[3])
datatup_full = files_to_array(allfiles[1], allfiles[3])
datatup_extranoise = files_to_array(allfiles[2], allfiles[3])
data_missing = datatup_missing[0]
data_full = datatup_full[0]
data_extranoise = datatup_extranoise[0]
gammalist_missing = datatup_missing[1]
gammalist_full = datatup_full[1]
gammalist_extranoise = datatup_extranoise[1]

# Datasets
fig = plt.figure()
ax = fig.gca()

for n, data, gammalist, c in zip(names, (data_full, data_missing, data_extranoise), (gammalist_full, gammalist_missing, gammalist_extranoise), colors ):
    try:
        int(n) + 1
        n = "No noise"
    except ValueError:
        if n == 'noise_10^-2':
            n = r"$\sigma$ = 0.01"
        else:
            n = r"$\sigma$ = 0.1"
        pass
    ax.plot(data[:,1], data[:,2], lw=2, marker='.', markersize=6, label=n, c=c)
ax.axhline(1.6408, ls='--', c='k')
# Set some plot limits/labels
ax.set_xlim(10**-8,1.0) # MSD 10^-8 -> 10^-2 = RMSE 10^-4 -> 10^-1
ax.set_xscale('log')
ax.set_ylim(0.0,20.0)
ax.set_ylabel(r"W$_{app}$ / kJ mol$^{-1}$")
ax.set_xlabel("MSD to target data")
ax.legend(frameon=False, loc='upper left')


#fig.suptitle("TeaA neutral trajectory,\nall frames, reweighting to mixed 60-40 data")
plt.tight_layout()
plt.savefig("Lcurve_allnoise.pdf", dpi=300)
