#!/usr/bin/env python

# Plots RMSFs of clusters from resfracs reweighting, gamma=1000

import cbf_colors
import numpy as np
import matplotlib.pyplot as plt


### Define defaults for matplotlib plots
plt.style.use('../styles+swatches/cbf_8color_cycler_paper.mplstyle')
###

# Dataset 1
fig = plt.figure()
ax = fig.gca()
for clust in range(-1,2):
    data = np.loadtxt("./data_folder/resfracs/bb_fluct_cluster_%d_eps4.22e+01.dat" % clust)
    if clust == -1:
        ax.plot(data[:,0].astype(np.int16), data[:,1], lw=2.5, ls='--', label="Outliers")
    else:
        ax.plot(data[:,0].astype(np.int16), data[:,1], lw=2.5, label="Cluster %d" % (clust+1))

# Set some plot limits/labels
xtickvals = [1]
xtickvals.extend(list(range(50,data[:,0].astype(np.int16)[-1]-50,50)))
xtickvals.append(data[:,0].astype(np.int16)[-1])
ax.set_xticks(xtickvals)
ax.set_xticklabels(xtickvals)
ax.set_xlim(data[0,0], data[-1,0])
ax.set_ylim(0.0, 2.5)
ax.set_ylabel(r"RMSF / $\AA$")
ax.set_xlabel("Residue")
ax.set_title("Backbone RMSF by residue\nafter ensemble refinement to residue-level target data")



# Final tidy
ax.legend(frameon=False, loc='lower right')
plt.tight_layout()
plt.savefig("RMSF_resfracs_gamma_1e+03.pdf", dpi=300)

