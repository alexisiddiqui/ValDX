#!/usr/bin/env python


# Plots weights applied to all trajectories along with moving average 

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import cbf_colors
from glob import glob
from scipy.stats import linregress as linefit

### Define defaults for matplotlib plots
plt.style.use('../styles+swatches/cbf_broryl_cycler_paper.mplstyle')
###

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
coverages = [ 100, 80, 60, 40, 20 ]
folder  = './data_folder/10/reduced_coverage/10^-6_MSD'
sorted_weightsfiles = [ './data_folder/10/10^-6_MSD/mixed_gamma_7.8x10^2_final_weights.dat' ]
sorted_workfiles = [ './data_folder/10/10^-6_MSD/mixed_gamma_7.8x10^2_work.dat' ]
sorted_weightsfiles.extend([ glob(os.path.join(folder, "*_cov_%s_final_weights.dat" % c ))[0] for c in coverages[1:] ])
sorted_workfiles.extend([ glob(os.path.join(folder, "*_cov_%s_work.dat" % c))[0] for c in coverages[1:] ])

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
rmsds = np.loadtxt("../fig3/data_folder/closed_rmsd_all.xvg", skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )
# Read in RMSDs of closed and extract indices of frames with RMSD < 1
openrmsds = np.loadtxt("../fig3/data_folder/open_rmsd_all.xvg", skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )
# Read in RMSDs of all and extract indices of frames with RMSD < 1.1, 1.25, 1.5, 1.75, 2.0
fullrmsds = np.loadtxt("../fig3/data_folder/neutral_rmsd_all.xvg", skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )
# Read in RMSDs of all to open and extract indices of frames with RMSD < 1.1, 1.25, 1.5, 1.75, 2.0
openfullrmsds = np.loadtxt("../fig3/data_folder/neutral_rmsd_open_all.xvg", skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )

idxlist = [ rmsds['idxs'].flatten() ] # These are the initial closed frames
openidxlist = [ openrmsds['idxs'].flatten() ] # These are the initial closed frames
rmsd_cuts = [ 0.11, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25 ]
for cut in rmsd_cuts:
    idxlist.append(np.where(fullrmsds['rmsd'] < cut)[0])
    openidxlist.append(np.where(openfullrmsds['rmsd'] < cut)[0])

# Use extracted indices to extract weights
closedlist = [ data[:,i] for i in idxlist ]
notclosedlist = [ np.delete(data, i, axis=1) for i in idxlist ]
openlist = [ data[:,i] for i in openidxlist ]
notopenlist = [ np.delete(data, i, axis=1) for i in openidxlist ]


# Plot
# Weighted RMSD distribution
fig = plt.figure()
ax = fig.gca()
bins = np.arange(0.0,0.402,0.002)*10
n, _ = np.histogram(fullrmsds['rmsd'].flatten()*10, density=True, bins=bins, weights=iniweights)
binmidp = bins[:-1]+0.01
ax.axvline(1.0, ls='--', c='k', ymax=0.75)
ax.set_prop_cycle(None) # resets to that in mplstyle
init_line, = ax.plot(binmidp, n, label="No reweighting", lw=4)
covs_lines = []
for name, w in zip(coverages,data[1:]):
    n, _ = np.histogram(fullrmsds['rmsd'].flatten()*10, weights=w, density=True, bins=bins)
#    binmidp = bins[:-1]+0.001
    c, = ax.plot(binmidp, n, label="%s%%" % name, lw=4)
    covs_lines.append(c)
#    currslope, currintercept, currrval, currpval, currstderr = linefit(w, fullrmsds['rmsd'].flatten())
#    regxs, regys = create_regression_line(currslope, currintercept)
#    ax.plot(regxs, regys, label="y = %3.2fx + %3.2f, $R^2$ = %3.2f" % (currslope, currintercept, currrval**2))
# Open


# Titles etc.
#fig.suptitle("RMSD distributions before & after reweighting, effect of reducing sequence coverage")
#ax.set_title("RMSD to closed TeaA structure")
ax.set_xlabel(r"RMSD / $\AA$")
ax.set_xlim(0, 4.0)
#ax.set_xticks(range(len(gammas)))
#ax.set_xticklabels(gammas)
#axs[0].set_ylim(0,np.max(weights)*1.1)
ax.set_ylim(0,2.5)
ax.set_ylabel("Probability density")

leg_1 = ax.legend([init_line], ["No reweighting"], loc=(0.65,0.91), ncol=2)
fullnames = []
leg_2 = ax.legend(covs_lines, [ "%s%%" % c for c in coverages ], loc=(0.65,0.75), ncol=2)
for h in leg_1.legendHandles:
    h.set_linewidth(6)
for h in leg_2.legendHandles:
    h.set_linewidth(6)
ax.add_artist(leg_1)


plt.tight_layout()
plt.savefig("coverage_comparison_RMSDs_closedonly.pdf", dpi=300)

