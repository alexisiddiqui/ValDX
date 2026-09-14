#!/usr/bin/env python

# Plots RMSDs from neutral trajectory

import cbf_colors
import numpy as np
import matplotlib.pyplot as plt


### Define defaults for matplotlib plots
plt.style.use('../styles+swatches/cbf_8color_cycler_paper.mplstyle')
###

# Dataset 1
fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={ 'width_ratios' : [4,1] }) # gridspec_kw adjusts sizes
data = np.loadtxt("./data_folder/neutral_rmsd_all.dat")
data[:,0] /= 1000. # ps to ns
data[:,1] *= 10. # nm to A
axs[0].plot(data[:,0], data[:,1], lw=0.5)
bins = np.arange(0,4.0,0.04)
n, _bins = np.histogram(data[:,1], bins=bins)
axs[1].plot(n,bins[:-1]+0.02, lw=2.5) # Skip rightmost edge of bins


# Set some plot limits/labels
#xtickvals = [1]
#xtickvals.extend(list(range(5000,data[:,0].astype(np.int16)[-1]-5000,5000)))
#xtickvals.append(data[:,0].astype(np.int16)[-1])
#ax.set_xticks(xtickvals)
#ax.set_xticklabels(xtickvals)
axs[0].set_xlim(data[0,0], data[-1,0])
axs[0].set_ylim(0.0, 4.0)
axs[0].set_ylabel(r"C$_{\alpha}$ RMSD / $\AA$")
axs[0].set_xlabel("Time / ns")
axs[1].set_xlabel("Frequency")
fig.suptitle(r"C$_{\alpha}$ RMSD in initial unbiased metadynamics ensemble"+"\nReferenced to closed TeaA structure")



# Final tidy
#plt.tight_layout()
plt.savefig("RMSD_neutral_traj.pdf", dpi=300)

