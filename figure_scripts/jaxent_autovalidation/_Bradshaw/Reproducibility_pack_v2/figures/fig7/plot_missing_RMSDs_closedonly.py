#!/usr/bin/env python


# Plots weights applied to all trajectories along with moving average 

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import cbf_colors
from glob import glob

### Define defaults for matplotlib plots
plt.style.use('../styles+swatches/cbf_8color_cycler_paper.mplstyle')


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
missinginiweights = np.ones(9199)
names = [ '10', '10_missing' ]
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

data = files_to_array(sorted_weightsfiles[:-1])
work = files_to_array(sorted_workfiles[:-1])[:,2]

missingdata = files_to_array([ sorted_weightsfiles[-1] ])
missingwork = files_to_array([ sorted_workfiles[-1] ])[:,2]

# Check sum of weights is correct
for d in data:
    print("Weights sum to 1?", np.isclose(np.sum(d),1.))

# Prepend initial weights
data = np.concatenate((np.ones((1,44494))/44494, data))
work = np.concatenate((np.array([0]),work))

missingdata = np.concatenate((np.ones((1,9199))/9199, missingdata))
missingwork = np.concatenate((np.array([0]),missingwork))

# Read in RMSDs of closed and extract indices of frames with RMSD < 1
rmsds = np.loadtxt(os.path.expandvars("./data_folder/10/closed_rmsd_all.xvg"), skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )
# Read in RMSDs of closed and extract indices of frames with RMSD < 1
openrmsds = np.loadtxt(os.path.expandvars("./data_folder/10/open_rmsd_all.xvg"), skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )
# Read in RMSDs of all and extract indices of frames with RMSD < 1.1, 1.25, 1.5, 1.75, 2.0
fullrmsds = np.loadtxt(os.path.expandvars("./data_folder/10/neutral_rmsd_all.xvg"), skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )
# Read in RMSDs of all to open and extract indices of frames with RMSD < 1.1, 1.25, 1.5, 1.75, 2.0
openfullrmsds = np.loadtxt(os.path.expandvars("./data_folder/10/neutral_rmsd_open_all.xvg"), skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )

idxlist = [ rmsds['idxs'].flatten() ] # These are the initial closed frames
openidxlist = [ openrmsds['idxs'].flatten() ] # These are the initial closed frames

# Read in RMSDs of all and extract indices of frames with RMSD < 1.1, 1.25, 1.5, 1.75, 2.0
missingfullrmsds = np.loadtxt("./data_folder/10_missing/gt1.5A_to_closed_rmsd_all.xvg", skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )
# Read in RMSDs of all to open and extract indices of frames with RMSD < 1.1, 1.25, 1.5, 1.75, 2.0
missingopenfullrmsds = np.loadtxt("./data_folder/10_missing/gt1.5A_to_closed_rmsd_toopen_all.xvg", skiprows=16, dtype=[ ('idxs',np.int32,(1,)) , ('rmsd',np.float64,(1,)) ] )



# Plot
# Weighted RMSD distribution
fig = plt.figure()
ax = fig.gca()
bins = np.arange(0.0,0.402,0.002)*10
n, _ = np.histogram(fullrmsds['rmsd'].flatten()*10, density=True, bins=bins, weights=iniweights)
binmidp = bins[:-1]+0.01
#ax.axvline(1.0, ls='--', c='k', ymax=0.75)
# Plot the full reference ensemble
init_line1, = ax.plot(binmidp, n, label="Initial - all structures", lw=4, c='cbf_sky_blue')
# Plot the truncated reference ensemble
bins = np.arange(0.0,0.402,0.002)*10
n, _ = np.histogram(missingfullrmsds['rmsd'].flatten()*10, density=True, bins=bins, weights=missinginiweights)
binmidp = bins[:-1]+0.01
init_line2, = ax.plot(binmidp, n, label="Initial - no closed structures", lw=4, c='cbf_blue')
ax.set_prop_cycle(None) # resets to that in mplstyle
segs_lines = []
for name, w in zip(names[:-1],data[1:]):
    name = name + " mixture"
    n, _ = np.histogram(fullrmsds['rmsd'].flatten()*10, weights=w, density=True, bins=bins)
#    binmidp = bins[:-1]+0.001
    s, = ax.plot(binmidp, n, label=name, lw=4)
    segs_lines.append(s)

for name, w in zip([ names[-1] ],[ missingdata[1,:] ]):
    name = name + " mixture"
    n, _ = np.histogram(missingfullrmsds['rmsd'].flatten()*10, weights=w, density=True, bins=bins)
#    binmidp = bins[:-1]+0.001
    s, = ax.plot(binmidp, n, label=name, lw=4)
    segs_lines.append(s)


# Titles etc.
#fig.suptitle("RMSD distributions before & after reweighting, effect of population averaging")
#ax.set_title("RMSD to closed TeaA structure")
ax.set_xlabel(r"RMSD / $\AA$")
ax.set_xlim(0, 4.0)
#ax.set_xticks(range(len(gammas)))
#ax.set_xticklabels(gammas)
#axs[0].set_ylim(0,np.max(weights)*1.1)
ax.set_ylim(0,4.0)
ax.set_ylabel("Probability density")
#axs[0].legend()
leg_1 = ax.legend([init_line1, init_line2], ["Initial - all structures", "Initial - no closed structures"], loc=(0.57,0.91), ncol=1)
fullnames = []
for name in names:
    try:
        int(name) + 1
        fullnames.append("Reweighted - all structures")
    except ValueError:
        fullnames.append("Reweighted - no closed structures")
        pass
leg_2 = ax.legend(segs_lines, fullnames, loc=(0.57,0.815), ncol=1)
for h in leg_1.legendHandles:
    h.set_linewidth(6)
for h in leg_2.legendHandles:
    h.set_linewidth(6)
ax.add_artist(leg_1)


plt.tight_layout()
plt.savefig("missing_comparison_RMSDs_closedonly.pdf", dpi=300)

