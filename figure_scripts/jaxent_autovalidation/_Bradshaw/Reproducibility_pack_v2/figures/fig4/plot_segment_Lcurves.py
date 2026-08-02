#!/usr/bin/env python

# Plots distributions of final weights across TeaA mixed trajectory frames

import cbf_colors
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import os

### Define defaults for matplotlib plots
plt.style.use('../styles+swatches/cbf_broryl_cycler_paper_80mm.mplstyle')
###

# Plot L-curve
names = [ 'residues', '5', '10', '15', '20', '50' ]
folders = [ './data_folder/' + i for i in names ]
powers = [ -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def make_filelist(folder, powers):
    files1 = []
    for p in powers:
        for j in range(1,10):
            files1.append(os.path.join(folder, "mixed_gamma_%sx10^%s_work.dat" % (j,p)))
    return files1
         
def files_to_array(fnames):
    """Read in data fom list of files with np.loadtxt.
       Returns array of shape (n_files, n_data_per_file)
       Empty files are dropped"""
    l = []
    for f in fnames:
        try:
            _ = np.loadtxt(f)
        except IOError:
            continue
        if _.shape == (0,):
            continue
        else:
            l.append(_)
    try:
        return np.stack(l, axis=0)
    except ValueError:
        raise ValueError("Error in stacking files read with np.loadtxt - are they all the same length?")

allfiles = [ make_filelist(f, powers) for f in folders ]
data = map(files_to_array, allfiles)


# Datasets
fig = plt.figure()
ax = fig.gca()
for d, n in zip(data,names):
    try:
        int(n) + 1
        n = n + "-residue"
    except ValueError:
        n = "Individual"
        pass
    ax.plot(d[:,1], d[:,2], lw=2, marker='.', markersize=6, label=n)
ax.axvline(10**-6, ymax=0.95, ls='--', c='k')
# Set some plot limits/labels
ax.set_xlim(10**-8,0.01) # MSD 10^-8 -> 10^-2 = RMSE 10^-4 -> 10^-1
ax.set_xscale('log')
ax.set_ylim(0.0,3.0)
ax.set_ylabel(r"Apparent work / kJ mol$^{-1}$")
ax.set_xlabel("MSD to target data")
#axes[2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#ax.set_title("Apparent work vs. MSD plot\nEffect of segment averaging")
ax.legend(frameon=False, loc=(0.7,0.45))


#fig.suptitle("TeaA neutral trajectory,\nall frames, reweighting to mixed 60-40 data")
plt.tight_layout()
plt.savefig("segment_comparison_Lcurves.pdf", dpi=300)
