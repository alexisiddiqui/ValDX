#!/usr/bin/env python

# Script to sum segment averages and take diffs Out - In 


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import cbf_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


### Define defaults for matplotlib plots
plt.style.use('../styles+swatches/cbf_broryl_minicycler_paper.mplstyle')
###

# Helper function to annotate every helix/sheet individually
def do_annotation_full(currax, yval=-0.1, height=0.08, divisor=1.):
    """Annotate the current axis with the domain definitions at:
       y = yval (default = -0.1), height = height (default 0.08)
       Optionally provide a divisor for the conversion of residue
       numbers (by which domains are defined) into the x-axis frame for
       the current plot. This is useful if e.g. your x-axis is actually
       individual segments. Default divisor = 1."""

    yval = yval - height/2  # Rectangle dimensions defined from bottom left
    # Store rectangle sizes as {(x, y, width, height, colour) : label}
    domainstart = np.asarray([1,13,32,47,58,64,86,112,141,148,160,166,180,186,199,206,216,225,264,271,294], dtype=np.float64)
    domainend = np.asarray([7,30,38,55,62,80,111,129,145,159,165,177,185,197,205,215,223,261,269,292,310], dtype=np.float64)
    domainwidth = domainend - domainstart
    domainstart /= divisor
    domainwidth /= divisor
    labels = { (domainstart[0], yval, domainwidth[0], height, 'cbf_softblue') : r"$\beta$1",
               (domainstart[1], yval, domainwidth[1], height, 'cbf_softblue') : r"$\alpha$1",
               (domainstart[2], yval, domainwidth[2], height, 'cbf_softblue') : r"$\beta$2",
               (domainstart[3], yval, domainwidth[3], height, 'cbf_softblue') : r"$\alpha$2",
               (domainstart[4], yval, domainwidth[4], height, 'cbf_softblue') : r"$\beta$3",
               (domainstart[5], yval, domainwidth[5], height, 'cbf_softblue') : r"$\alpha$3",
               (domainstart[6], yval, domainwidth[6], height, 'cbf_softblue') : r"$\alpha$4",
               (domainstart[7], yval, domainwidth[7], height, 'cbf_peach') : r"$\beta$4",
               (domainstart[8], yval, domainwidth[8], height, 'cbf_redrusset') : r"$\beta$5",
               (domainstart[9], yval, domainwidth[9], height, 'cbf_redrusset') : r"$\alpha$5",
               (domainstart[10], yval, domainwidth[10], height, 'cbf_redrusset') : r"$\beta$6",
               (domainstart[11], yval, domainwidth[11], height, 'cbf_redrusset') : r"$\alpha$6",
               (domainstart[12], yval, domainwidth[12], height, 'cbf_redrusset') : r"$\beta$7",
               (domainstart[13], yval, domainwidth[13], height, 'cbf_redrusset') : r"$\alpha$7",
               (domainstart[14], yval, domainwidth[14], height, 'cbf_redrusset') : r"$\beta$8",
               (domainstart[15], yval, domainwidth[15], height, 'cbf_softblue') : r"$\beta$9",
               (domainstart[16], yval, domainwidth[16], height, 'cbf_softblue') : r"$\alpha$8", 
               (domainstart[17], yval, domainwidth[17], height, 'cbf_peach') : r"$\alpha$9", 
               (domainstart[18], yval, domainwidth[18], height, 'cbf_redrusset') : r"$\beta$10", 
               (domainstart[19], yval, domainwidth[19], height, 'cbf_redrusset') : r"$\alpha$10", 
               (domainstart[20], yval, domainwidth[20], height, 'cbf_redrusset') : r"$\alpha$11" }
    for key, l in labels.items():
        xmin, ymin, w, h, c = key
        currrect = currax.add_artist(mpatch.FancyBboxPatch((xmin,ymin), w, h, color=c, boxstyle="round, pad=0.0, rounding_size=5", joinstyle="round", capstyle='round', lw=None, mutation_aspect=2.5/310.))
        rx, ry = currrect.get_x(), currrect.get_y()
        cx = rx + currrect.get_width()/2.
        cy = ry + currrect.get_height()/2.
        currax.annotate(l, (cx, cy), color='w', fontsize=14, weight='normal', ha='center', va='center', rotation='vertical') 

# Helper function to annotate just the Nter/Cter/hinge domains
def do_annotation_domains(currax, yval=-0.1, height=0.08, divisor=1.):
    """Annotate the current axis with the domain definitions at:
       y = yval (default = -0.1), height = height (default 0.08)
       Optionally provide a divisor for the conversion of residue
       numbers (by which domains are defined) into the x-axis frame for
       the current plot. This is useful if e.g. your x-axis is actually
       individual segments. Default divisor = 1."""

    yval = yval - height/2  # Rectangle dimensions defined from bottom left
    # Store rectangle sizes as {(x, y, width, height, colour) : label}
    domainstart = np.asarray([2,112,141,206,225,264], dtype=np.float64)
    domainend = np.asarray([111,129,205,223,261,310], dtype=np.float64)
    domainwidth = domainend - domainstart
    domainstart /= divisor
    domainwidth /= divisor
    labels = { (domainstart[0], yval, domainwidth[0], height, 'cbf_softblue') : "N",
               (domainstart[1], yval, domainwidth[1], height, 'cbf_peach') : r"$\beta$4",
               (domainstart[2], yval, domainwidth[2], height, 'cbf_redrusset') : "C",
               (domainstart[3], yval, domainwidth[3], height, 'cbf_softblue') : "N",
               (domainstart[4], yval, domainwidth[4], height, 'cbf_peach') : r"$\alpha$9", 
               (domainstart[5], yval, domainwidth[5], height, 'cbf_redrusset') : "C" }
    for key, l in labels.items():
        xmin, ymin, w, h, c = key
        currrect = currax.add_artist(mpatch.FancyBboxPatch((xmin,ymin), w, h, color=c, boxstyle="round, pad=0.0, rounding_size=5", joinstyle="round", capstyle='round', lw=None, mutation_aspect=1.01/294.))
        rx, ry = currrect.get_x(), currrect.get_y()
        cx = rx + currrect.get_width()/2.
        cy = ry + currrect.get_height()/2.
        currax.annotate(l, (cx, cy), color='w', fontsize=18, weight='heavy', ha='center', va='center', rotation=0) 

# Setup variables
times = [0.167, 1., 10., 60., 120.]
labels = ("closed", "open")

# The "%s-%s_diffs.dat" file contains the residue-based deuterated
# fraction differences and residue numbers and can be used to recreate
# the plot without the links to the original data.
results1 = np.loadtxt("%s-%s_diffs.dat" % labels, dtype=[('res', np.int32, (1,))], usecols=(0,))
diffs = np.loadtxt("%s-%s_diffs.dat" % labels, usecols=(1,2,3,4,5))

xs = results1['res'].flatten()

# Make heatmap 
fig, axs = plt.subplots(2,1)
xtickvals = [1]
xtickvals.extend(list(range(50,xs[-1]-50,50)))
xtickvals.append(xs[-1])
xticks = [0]
for i in xtickvals[1:]:
    xticks.append(np.where(xs==i)[0]) 
img = axs[0].imshow(diffs.T, cmap='RdBu', aspect=15.0, vmin=-1.0, vmax=1.0, interpolation='none')
axs[0].set_anchor('S')
axs[0].set_xticks(xticks)
axs[0].set_xticklabels(xtickvals)
axs[0].set_yticklabels(times)
axs[0].set_yticks(list(range(len(times))))
axs[0].set_ylabel("Time / min", weight='normal')
#axs[0].set_title("Difference in predicted deuterated fractions, %s - %s" % labels, y=1.12)
#axs[0].set_xlabel("Residue")

# Set size & location of colorbar
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.16)
cb_ax = fig.add_axes([0.83, 0.53, 0.02, 0.23])

#divider = make_axes_locatable(axs[0])
#cax0 = divider.append_axes("right", size="5%", pad=0.05)
#cax0.set_aspect('auto')
cbar = fig.colorbar(img, cax=cb_ax, ticks=[-1,0,1])
cbar.ax.set_ylabel(r"$\Delta$D$_i$ (closed - open)", rotation=270., labelpad=20, weight='normal', fontsize=18)
cbar.ax.tick_params(axis='both', which='both',length=0)
cbar.outline.remove()

# Make annotations
axs[1].axis('off')
axs[1].set_ylim(0,1.01)
axs[1].set_xlim(1,310.5)
axs[1].set_anchor('N')
do_annotation_domains(axs[1], yval=0.93, height=0.14)

plt.savefig("Differences_%s-%s_heatmap_RdBu_filled.png" % labels, dpi=300)
