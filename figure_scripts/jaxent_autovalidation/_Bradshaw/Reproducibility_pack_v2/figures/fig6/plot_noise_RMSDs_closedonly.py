#!/usr/bin/env python


# Plots weights applied to all trajectories along with moving average 

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import cbf_colors
from glob import glob
from scipy.stats import linregress as linefit

### Define defaults for matplotlib plots
plt.style.use('../styles+swatches/cbf_8color_cycler_paper.mplstyle')


colors = [ 'cbf_key', 'cbf_orange', 'cbf_bluish_green' ]

def moving_average(vals, window=3) :
    ma = np.cumsum(vals, dtype=float)
    ma[window:] = ma[window:] - ma[:-window]
    return ma[window - 1:] / window

# Regression line generation
def create_regression_line(slope, intercept, vmin=0., vmax=1., n=20.):
    xvals = np.linspace(vmin, vmax, n)
    yvals = xvals*slope + intercept
    return xvals, yvals

iniweights = np.ones(44494)
names = [ '10', 'noise_10^-2', 'noise_10^-1' ]
folders = [ './data_folder/' + i + '/fit_to_work' for i in names ]
sorted_weightsfiles = [ glob(os.path.join(f, "*final_weights.dat"))[0] for f in folders ]
sorted_workfiles = [ glob(os.path.join(f, "*work.dat"))[0] for f in folders ]

def files_to_array(fnames):
    """Read in data fom list of files with np.loadtxt.
       Returns array of shape (n_files, n_data_per_file)
       Empty files are dropped"""
    l = []
    for f in fnames:
        _ = np.loadtxt(f)
        if _.shape == (0,):
            continue
        else:
            l.append(_)
    try:
        return np.stack(l, axis=0)
    except ValueError:
        raise ValueError("Error in stacking files read with np.loadtxt - are they all the same length?")

data = files_to_array(sorted_weightsfiles)
work = files_to_array(sorted_workfiles)[:,2]

# Check sum of weights is correct
for d in data:
    print("Weights sum to 1?", np.isclose(np.sum(d),1.))

# Prepend initial weights
data = np.concatenate((np.ones((1,44494))/44494, data))
work = np.concatenate((np.array([0]),work))

# Read in RMSDs of closed and extract indices of frames with RMSD < 1
rmsds = np.loadtxt(os.path.expandvars("../fig3/data_folder/closed_rmsd_all.xvg"), skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )
# Read in RMSDs of closed and extract indices of frames with RMSD < 1
openrmsds = np.loadtxt(os.path.expandvars("../fig3/data_folder/open_rmsd_all.xvg"), skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )
# Read in RMSDs of all and extract indices of frames with RMSD < 1.1, 1.25, 1.5, 1.75, 2.0
fullrmsds = np.loadtxt(os.path.expandvars("../fig3/data_folder/neutral_rmsd_all.xvg"), skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )
# Read in RMSDs of all to open and extract indices of frames with RMSD < 1.1, 1.25, 1.5, 1.75, 2.0
openfullrmsds = np.loadtxt(os.path.expandvars("../fig3/data_folder/neutral_rmsd_open_all.xvg"), skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )

idxlist = [ rmsds['idxs'].flatten() ] # These are the initial closed frames
openidxlist = [ openrmsds['idxs'].flatten() ] # These are the initial closed frames


# Plot
# Weighted RMSD distribution
fig = plt.figure()
ax = fig.gca()
bins = np.arange(0.0,0.402,0.002)*10
n, _ = np.histogram(fullrmsds['rmsd'].flatten()*10, density=True, bins=bins, weights=iniweights)
binmidp = bins[:-1]+0.01
#ax.axvline(1.0, ls='--', c='k', ymax=0.75)
# Initial no reweighting distribution
init_line, = ax.plot(binmidp, n, label="No reweighting", c='cbf_sky_blue', lw=4)
segs_lines = []
for name, w, c in zip(names, data[1:], colors):
    name = name + " mixture"
    n, _ = np.histogram(fullrmsds['rmsd'].flatten()*10, weights=w, density=True, bins=bins)
#    binmidp = bins[:-1]+0.001
    s, = ax.plot(binmidp, n, label=name, c=c, lw=4)
    segs_lines.append(s)

# Titles etc.
ax.set_xlabel(r"RMSD to closed / $\AA$")
ax.set_xlim(0, 4.0)
#ax.set_xticks(range(len(gammas)))
#ax.set_xticklabels(gammas)
#axs[0].set_ylim(0,np.max(weights)*1.1)
ax.set_ylim(0,2.5)
ax.set_ylabel("Probability density")
#axs[0].legend()
leg_1 = ax.legend([init_line], ["No reweighting"], loc=(0.70,0.91), ncol=1)
fullnames = []
for name in names:
    try:
        int(name) + 1
        fullnames.append("No noise")
    except ValueError:
        if name == 'noise_10^-2':
            fullnames.append(r"$\sigma$ = 0.01")
        else:
            fullnames.append(r"$\sigma$ = 0.1")
        pass
leg_2 = ax.legend(segs_lines, fullnames, loc=(0.70,0.78), ncol=1)
for h in leg_1.legendHandles:
    h.set_linewidth(6)
for h in leg_2.legendHandles:
    h.set_linewidth(6)
ax.add_artist(leg_1)


plt.tight_layout()
plt.savefig("noise_comparison_RMSDs_closedonly.pdf", dpi=300)

