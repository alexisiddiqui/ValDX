#!/usr/bin/env python

# Plots L-curve of gamma vs MSD for the results of reweighting with individual residues

import cbf_colors
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import os

### Define defaults for matplotlib plots
plt.style.use('../styles+swatches/cbf_broryl_cycler_paper_80mm.mplstyle')


# Plot L-curve
names = [ 'residues' ]
folders = [ './data_folder/' + i for i in names ]
powers = [ -2, -1, 0, 1, 2, 3 ]

def make_filelist(folder, powers):
    files1 = []
    files2 = []
    gammas = []
    for p in powers:
        for j in range(1,10):
            files1.append(os.path.join(folder, "mixed_gamma_%sx10^%s_work.dat" % (j,p)))
            files2.append(os.path.join(folder, "mixed_gamma_%sx10^%s_per_iteration_output.dat" % (j,p)))
            gammas.append(j * 10**p)
    return files1, gammas, files2
         
def files_to_array(fnames, gammas, iterfiles):
    """Read in data fom list of files with np.loadtxt.
       Returns array of shape (n_files, n_data_per_file)
       Empty files are dropped"""
    l = []
    foundgammas = []
    founditerfiles = []
    for f, g, i in zip(fnames, gammas, iterfiles):
        try:
            _ = np.loadtxt(f)
        except IOError:
            continue
        if _.shape == (0,):
            continue
        else:
            l.append(_)
            foundgammas.append(g)
            founditerfiles.append(i)
    try:
        return np.stack(l, axis=0), foundgammas, founditerfiles
    except ValueError:
        raise ValueError("Error in stacking files read with np.loadtxt - are they all the same length?")

allfiles = make_filelist(folders[0], powers)
iterfiles = allfiles[2]
datatup = files_to_array(allfiles[0], allfiles[1], allfiles[2])
data = datatup[0]
gammalist = datatup[1]
iterfilelist = datatup[2]

bhs = []
bcs = []
for itfn in iterfilelist:
    last = subprocess.check_output(["tail", "-1", "%s" % itfn ])
    bhs.append(last.split()[6])
    bcs.append(last.split()[7])

bhs = np.array(list(map(float, bhs)))
bcs = np.array(list(map(float, bcs)))

# Datasets
fig, axs = plt.subplots(2,2, figsize=(3.1496*2, 2.4338*2))

for n in names:
    try:
        int(n) + 1
        n = n + "-residue"
    except ValueError:
        n = "Individual residue\nreweighting"
        pass
    axs[0,0].plot(gammalist, data[:,1], lw=2, marker='.', markersize=6, label=n, color='k')
    axs[0,1].plot(data[:,1], data[:,2], lw=2, marker='.', markersize=6, label=n, color='k')
    axs[1,0].plot(gammalist, bhs, lw=2, marker='.', markersize=6, label=r"$\beta_{H}$", color='k')
    axs[1,0].plot(gammalist, bcs, lw=2, marker='.', markersize=6, label=r"$\beta_{C}$", color='cbf_sky_blue')
    axs[1,1].plot(data[:,1], bhs, lw=2, marker='.', markersize=6, label=r"$\beta_{H}$", color='k')
    axs[1,1].plot(data[:,1], bcs, lw=2, marker='.', markersize=6, label=r"$\beta_{C}$", color='cbf_sky_blue')
#ax.axvline(10**-6, ymax=0.95, ls='--', c='k')
# Set some plot limits/labels
axs[0,0].set_ylim(10**-8,0.01) # MSD 10^-8 -> 10^-2 = RMSE 10^-4 -> 10^-1
axs[0,0].set_yscale('log')
axs[0,0].set_xlim(0.001,10**3)
axs[0,0].set_xscale('log')
axs[0,0].set_xlabel(r"$\gamma$")
axs[0,0].set_ylabel("MSD to target data")
axs[0,1].set_xlim(10**-8,0.01) # MSD 10^-8 -> 10^-2 = RMSE 10^-4 -> 10^-1
axs[0,1].set_xscale('log')
axs[0,1].set_ylim(0.0,3.0)
axs[0,1].set_ylabel(r"W$_{app}$ / kJ mol$^{-1}$")
axs[0,1].set_xlabel("MSD to target data")
axs[1,0].set_ylim(0, 3.0) # MSD 10^-8 -> 10^-2 = RMSE 10^-4 -> 10^-1
axs[1,0].set_xlim(0.001,10**3)
axs[1,0].set_xscale('log')
axs[1,0].set_xlabel(r"$\gamma$")
axs[1,0].set_ylabel(r"$\beta$ parameter")
axs[1,0].set_ylim(0.0,3.0)
axs[1,1].set_xlim(10**-8,0.01) # MSD 10^-8 -> 10^-2 = RMSE 10^-4 -> 10^-1
axs[1,1].set_xscale('log')
axs[1,1].set_ylim(0.0,3.0)
axs[1,1].set_ylabel(r"$\beta$ parameter")
axs[1,1].set_xlabel("MSD to target data")
#axes[2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#ax.set_title("Apparent work vs. MSD plot\nEffect of segment averaging")
axs[0,1].legend(frameon=False, loc='upper right')
axs[1,1].legend(frameon=False, loc='upper right')


#fig.suptitle("TeaA neutral trajectory,\nall frames, reweighting to mixed 60-40 data")
axs[0,0].minorticks_off()
axs[1,0].minorticks_off()
plt.tight_layout()
plt.savefig("Decision_plots+betas.pdf", dpi=300)
