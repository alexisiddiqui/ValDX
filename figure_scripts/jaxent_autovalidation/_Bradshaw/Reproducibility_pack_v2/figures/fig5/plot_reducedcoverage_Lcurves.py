#!/usr/bin/env python

# Plots L-curves of 10-residue segment data at different % coverage

import cbf_colors
import numpy as np
import matplotlib.pyplot as plt
import sys
from cycler import cycler
from glob import glob

### Define defaults for matplotlib plots
plt.style.use('../styles+swatches/cbf_broryl_cycler_paper_80mm.mplstyle')
###


coverages = [ 80, 60, 40, 20 ]

powers = [ -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# Plot first L-curve 
files1 = []
gammas = []
for p in powers:
    for j in range(1,10):
        files1.append("data_folder/10/mixed_gamma_%sx10^%s_work.dat" % (j,p))
        gammas.append(j*(10**p))


def files_to_array(fnames, gammas):
    """Read in data fom list of files with np.loadtxt.
       Returns array of shape (n_files, n_data_per_file)
       Empty files are dropped"""
    l = []
    gl = []
    for f, g in zip(fnames,gammas):
        try:
            _ = np.loadtxt(f)
        except IOError:
            continue
        if _.shape == (0,):
            continue
        else:
            l.append(_)
            gl.append(g)
    try:
        return np.stack(l, axis=0), np.array(gl)
    except ValueError:
        raise ValueError("Error in stacking files read with np.loadtxt - are they all the same length?")

# Dataset 1
fig = plt.figure()
ax = fig.gca()
data, _ = files_to_array(files1, gammas)

ax.plot(data[:,1], data[:,2], lw=2, marker='.', markersize=6, label="100%")
# Set some plot limits/labels
ax.set_xlim(10**-8,0.01)
ax.set_xscale('log')
ax.set_ylim(0.0,3.0)
ax.set_ylabel(r"Apparent work / kJ mol$^{-1}$")
ax.set_xlabel("MSD to target data")
#axes[2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#ax.set_title("Apparent work vs. MSD plot\nEffect of reducing sequence coverage")


# Plot remaining L-curves
for coverage in coverages:
    files1 = []
    gammas = []
    for p in powers:
        for j in range(1,10):
            files1.append("data_folder/10/reduced_coverage/mixed_gamma_%sx10^%s_cov_%s_work.dat" % (j,p,coverage))
            files1.append("data_folder/10/reduced_coverage/mixed_gamma_%sx10^%s_cov_%s_rst_work.dat" % (j,p,coverage)) # Sometimes the restarts are under separate names
            gammas.append(j*(10**p))
            gammas.append(j*(10**p)) # Add twice for the 2 files we've added above!
    data, _ = files_to_array(files1, gammas)
    ax.plot(data[:,1], data[:,2], lw=2, marker='.', markersize=6, label="%s%%" % coverage)

ax.axvline(10**-6, ymax=0.95, ls='--', c='k')
# Final tidy
ax.legend(frameon=False, loc='upper right')
plt.tight_layout()
plt.savefig("coverage_comparison_Lcurves.pdf", dpi=300)

