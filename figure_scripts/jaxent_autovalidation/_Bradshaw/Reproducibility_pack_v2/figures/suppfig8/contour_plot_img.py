#!/usr/bin/env python

# Script to make a contour plot of chi^2 values for each combo of
# Contacts and Hbonds values from *per_iteration_output files

import cbf_colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

### Define defaults for matplotlib plots
plt.style.use('../styles+swatches/cbf_broryl_cycler_paper.mplstyle')





# all_chisquareds.dat is created from `cat *per_iteration_output.dat > all_chisquareds.dat`
data = pd.read_csv("./data_folder/all_chisquareds.dat", delim_whitespace=True, comment="#", header=0)
# Sort descending from 12.9 to 1.0 for Bh, and ascending from 0.1 to 0.5 for Bc
data_sorted = data.sort_values(by=['Bh', 'Bc'], ascending=[False, True], axis='rows')

n_contacts = 41
n_hbonds = 121
xvals = np.linspace(0.1, 0.5, n_contacts)
yvals = np.linspace(13.0, 1.0, n_hbonds)
xvmesh, yvmesh = np.meshgrid(xvals, yvals)
# Reshape sorted data into grid where:
# x = increasing contact_beta
# y = decreasinf hbond_beta
z = data_sorted['chisquare'].values.reshape(int(len(data)/41),41)

# Contour levels
contours = np.array([0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
fig = plt.figure()
ax = fig.gca()

conplt = ax.contour(xvmesh, yvmesh, z, cmap='inferno', vmax=0.3, vmin=0.001, levels=contours, linewidths=4)
ax.clabel(conplt, inline=1)
img = ax.imshow(z, cmap='binary', interpolation=None, aspect='auto', extent=[0.1,0.5,1.0,13.0], vmax=0.4, vmin=0.00)
ax.set_xlabel(r"$\beta$$_C$")
ax.set_ylabel(r"$\beta$$_H$")
#ax.set_title(r"Contour plot of MSD values to expt. data" +\
#              "\nMixed 60/40 trajectory to mixed 60/40 data")

cbar = plt.colorbar(img)
cbar.set_label("MSD", rotation=270, labelpad=20, weight='normal')
plt.tight_layout()
plt.savefig("TeaA_mixed_contour_plot.pdf", dpi=300)

