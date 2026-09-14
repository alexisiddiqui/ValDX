#!/usr/bin/env python


# Plots weights applied to all trajectories along with moving average 

import sys
import numpy as np
import matplotlib.pyplot as plt
import cbf_colors
from cycler import cycler
from glob import glob

### Define defaults for matplotlib plots
plt.style.use('../styles+swatches/cbf_8color_cycler_paper_80mm.mplstyle')


def moving_average(vals, window=3) :
    ma = np.cumsum(vals, dtype=float)
    ma[window:] = ma[window:] - ma[:-window]
    return ma[window - 1:] / window

_wt_weightsfiles = glob("./data_folder/6traj_to_WT/6traj_to_WT_*final_weights.dat")
_mut_weightsfiles = glob("./data_folder/6traj_to_Y268A/6traj_to_Y268A_*final_weights.dat")
sorted_wtweightsfiles = []
sorted_wtworkfiles = []
sorted_mutweightsfiles = []
sorted_mutworkfiles = []
wtgammas = []
mutgammas = []
for e in range(-2,4):
    for n in range(1,10):
        s = str(n) + "x10^" + str(e)
        if "./data_folder/6traj_to_WT/6traj_to_WT_gamma_%s_final_weights.dat" % s in _wt_weightsfiles:
            sorted_wtweightsfiles.append("./data_folder/6traj_to_WT/6traj_to_WT_gamma_%s_final_weights.dat" % s)
            sorted_wtworkfiles.append("./data_folder/6traj_to_WT/6traj_to_WT_gamma_%s_work.dat" % s)
            wtgammas.append(n * 10**e)
        else:
            pass

for e in range(-2,4):
    for n in range(1,10):
        s = str(n) + "x10^" + str(e)
        if "./data_folder/6traj_to_Y268A/6traj_to_Y268A_gamma_%s_final_weights.dat" % s in _mut_weightsfiles:
            sorted_mutweightsfiles.append("./data_folder/6traj_to_Y268A/6traj_to_Y268A_gamma_%s_final_weights.dat" % s)
            sorted_mutworkfiles.append("./data_folder/6traj_to_Y268A/6traj_to_Y268A_gamma_%s_work.dat" % s)
            mutgammas.append(n * 10**e)
        else:
            pass



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

wtdata = files_to_array(sorted_wtweightsfiles)
wtwork = files_to_array(sorted_wtworkfiles)[:,2]
mutdata = files_to_array(sorted_mutweightsfiles)
mutwork = files_to_array(sorted_mutworkfiles)[:,2]

#data *= data.shape[1]
wtoutward = wtdata[:,:int(wtdata.shape[1]/2)]
wtinward = wtdata[:,int(wtdata.shape[1]/2):]
mutoutward = mutdata[:,:int(mutdata.shape[1]/2)]
mutinward = mutdata[:,int(mutdata.shape[1]/2):]
wtoutward = np.sum(wtoutward, axis=1)
wtinward = np.sum(wtinward, axis=1)
mutoutward = np.sum(mutoutward, axis=1)
mutinward = np.sum(mutinward, axis=1)

xs = np.arange(0.1, 10.1, 0.1)
interp_mut_inward = np.interp(xs, mutwork, mutinward)
interp_mut_outward = np.interp(xs, mutwork, mutoutward)
interp_wt_inward = np.interp(xs, wtwork, wtinward)
interp_wt_outward = np.interp(xs, wtwork, wtoutward)


# 0) Scatter plot 1) moving average plot
fig = plt.figure()
ax = fig.gca()
# wt/mutdiffs are the delta(in - out), i.e. the % excess of inward frames)
wtdiffs = interp_wt_inward - interp_wt_outward
mutdiffs = interp_mut_inward - interp_mut_outward
wtinvdiffs = interp_wt_outward - interp_wt_inward
mutinvdiffs = interp_mut_outward - interp_mut_inward
# Diffs are the deltadelta(in - out), i.e. the delta(% excess of inward frames) when fitting to inward, not outward)
diffs = mutdiffs - wtdiffs
invdiffs = mutinvdiffs - wtinvdiffs
ax.plot(xs, wtdiffs, label="WT inward-outward")
ax.plot(xs, mutdiffs, label="Y268A inward-outward")
ax.plot(xs, diffs, label=r"$\Delta$$\Delta$(inward-outward)", c='cbf_key', ls=':')


# Titles etc.
#fig.suptitle("Sum of weights from outward & inward facing simulations")
#ax.set_title("Reweighting only to WT data")
ax.set_xlabel(r"Work / kJ mol$^{-1}$")
ax.set_xlim(0, 10)
#ax.set_xscale('log')
#ax.set_xticks(range(len(gammas)))
#ax.set_xticklabels(gammas)
#axs[0].set_ylim(0,np.max(weights)*1.1)
ax.set_ylim(-1, 1)
ax.set_ylabel(r"$\Delta$(Fractional population)")
ax.axhline(0.0, ls='--', zorder=0)
ax.legend()



plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("6traj_Y268A-WT_deltas_interpolated_singleplot.pdf", dpi=300)

