#!/usr/bin/env python

# Script to calculate cluster membership

import mdtraj as md
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from cycler import cycler
from sklearn.cluster import DBSCAN as dbscan
from sklearn import metrics
from glob import glob

### Define defaults for matplotlib plots
plt.rc('lines', linewidth=3, markersize=4)
plt.rc('axes', prop_cycle=(cycler('color', ['k','b','r','orange','c','m','y','g'])), # Color cycle defaults to black
       labelweight='heavy', labelsize=22, titlesize=22) # Default fontsizes for printing
plt.rc('axes.spines', top=False, right=False) # Switch off top/right axes
plt.rc('legend', fontsize=16) # Default fontsizes for printing
plt.rc('xtick', labelsize=16) # Default fontsizes for printing
plt.rc('ytick', labelsize=16) # Default fontsizes for printing
plt.rc('figure', figsize=(11,8.5), titlesize=22, titleweight='heavy') # Default fontsizes for printing
#plt.rc('text', usetex=True)

# Import trajectory & calculate CA pairwise dists

trjdir = "../../data/trajectories"
top = md.load_topology(os.path.join(trjdir, "TeaA_ref_closed_state.pdb"))
ca = top.select("name CA")
traj = md.load(os.path.join(trjdir, "TeaA_initial_reimaged.xtc"), top=top, atom_indices=ca)

# First get the 2D CA RMSD
rmsd2d = np.zeros((traj.n_frames, traj.n_frames))
for i in range(traj.n_frames):
    rmsd2d[i] = md.rmsd(traj, traj, frame=i) * 10 # *10 to convert to Angstrom
    if (i+1) % 100 == 0:
        print("Done RMSD for frame %s of %s" % (i+1, traj.n_frames))
plt.imshow(rmsd2d, interpolation='none', vmin=0.0, vmax=4.0)
plt.colorbar(label=r"RMSD / $\AA$")
plt.savefig("2D_RMSD.png", dpi=300)

dists = metrics.pairwise.euclidean_distances(rmsd2d)

#maxdist, meandist = np.max(edists), np.mean(edists)
#maskarr = np.ma.masked_equal(edists, 0.0, copy=False)
#mindist = maskarr.min()
#del edists # for memory

#epsilons = np.linspace(mindist/2., maxdist, num=10)
#epsilons = np.linspace(1.0, 10.0, num=10)

# Choose epsilon to be an average of +/- cutoff A from every point
cutoffs = np.linspace(0.05, 0.5, 10)
epsilons = np.sqrt((cutoffs**2)*traj.n_frames)
clusters, noisepoints, silhouettes = [], [], []

for eps in epsilons:
    db = dbscan(eps=eps, min_samples=int(traj.n_frames/10.0), metric='precomputed', n_jobs=4).fit(dists)
    np.savetxt("Initial_cluster_labels_eps_%.2e.dat" % eps, db.labels_.T, \
               header="Full epsilon = %12.10f, min_samples=%s" % (eps, int(traj.n_frames/10.0)), fmt="%d")
    # metrics from dbscan tutorial 
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    # Only have silhouette score for unsupervised samples
    clusters.append(n_clusters_)
    noisepoints.append(n_noise_)
    try:
        silhouettes.append(metrics.silhouette_score(rmsd2d, labels))
    except ValueError: # e.g. only one cluster, len(labels) = 1
        silhouettes.append(0.0)
        pass
    print("Completed clustering with epsilon %8.5f" % eps)

clusters, noisepoints, silhouettes = np.array(clusters), np.array(noisepoints), np.array(silhouettes)
np.savetxt("All_N_clusters.dat", clusters.T, fmt="%d")
np.savetxt("All_N_noisepoints.dat", noisepoints.T, fmt="%d")
np.savetxt("All_silhouettes.dat", silhouettes.T, fmt="%7.5f")

besteps = epsilons[np.argmax(silhouettes)]
print("Initial best N_clusters = %s" % clusters[np.argmax(silhouettes)])
print("Initial best N_noise = %s" % noisepoints[np.argmax(silhouettes)])
print("Initial best silhouette score = %s" % silhouettes[np.argmax(silhouettes)])

weightsdir = "../../figures/fig3/data_folder"
_weightsfiles = glob(os.path.join(weightsdir, "mixed_gamma_*final_weights.dat"))
sorted_weightsfiles = []
gammas = []
for e in range(-1,4):
    for n in range(1,2):
        s = str(n) + "x10^" + str(e)
        if os.path.join(weightsdir, "mixed_gamma_%s_final_weights.dat" % s) in _weightsfiles:
            sorted_weightsfiles.append(os.path.join(weightsdir, "mixed_gamma_%s_final_weights.dat" % s))
            gammas.append(n * 10**e)
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

frameweights = files_to_array(sorted_weightsfiles)*traj.n_frames
colordict = { -1 : 'k', 0 : 'b', 1 : 'r', 2 : 'orange'}
for besteps in epsilons:
    for g, w in zip(gammas, frameweights):
        db = dbscan(eps=besteps, min_samples=int(traj.n_frames/10.0), metric='precomputed', n_jobs=4).fit(dists, sample_weight=w)
        np.savetxt("Gamma_%.e_cluster_labels_eps_%.2e.dat" % (g, besteps), \
                   db.labels_.T, \
                   header="Full epsilon = %12.10f, min_samples=%s" % (besteps, int(traj.n_frames/10.0)), fmt="%d")
        # metrics from dbscan tutorial 
        labels = db.labels_
        outlierfilter = (labels != -1)
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        # Only have silhouette score for unsupervised samples
        with open("Gamma_%.e_stats.dat" % g, 'a') as f:
            f.write("Epsilon = %.5e\n" % besteps)
            f.write("N_clusters = %d\n" % n_clusters_)
            f.write("N_noise_points = %d\n" % n_noise_)
            try:
                # Filter dists to only give silhouette scores of REAL clusters, not outliers
                silhouettescore = metrics.silhouette_score(dists[outlierfilter,:][:,outlierfilter], labels[outlierfilter], metric='precomputed')
                f.write("Silhouette_score = %7.5f\n" % silhouettescore)
                silhouettevals = metrics.silhouette_samples(dists[outlierfilter,:][:,outlierfilter], labels[outlierfilter], metric='precomputed')
                fig = plt.figure()
                ax = fig.gca()
                ax.set_xlim([-0.1, 1])
                ax.set_ylim([0, len(dists[outlierfilter,:][:,outlierfilter]) + (n_clusters_ + 1) * 10])
                y_lower = 10
                for i in range(n_clusters_):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = silhouettevals[labels[outlierfilter] == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = colordict[i]
                    ax.fill_betweenx(np.arange(y_lower, y_upper),
                                     0, ith_cluster_silhouette_values,
                                     facecolor=color, edgecolor=color, alpha=0.7)

                    # Label the silhouette plots with their cluster numbers at the middle
                    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1), fontsize=22)

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax.set_title("Silhouette plot for clusters at epsilon %.2e" % besteps)
                ax.set_xlabel("Silhouette coefficient values")
                ax.set_ylabel("Cluster number")

                # The vertical line for average silhouette score of all the values
                ax.axvline(x=silhouettescore, color="k", linestyle="--")
                ax.set_yticks([])  # Clear the yaxis labels / ticks
                ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                ax.spines['left'].set_visible(False)
                plt.savefig("Silhouette_scores_eps_%.2e.png" % besteps, dpi=300)
            except ValueError: # e.g. only one cluster, len(labels) = 1
                f.write("Silhouette_score = 0.0 (only single cluster)\n")

# Plotting/analysis of cluster results
    fn = "Gamma_1e+03_cluster_labels_eps_%.2e.dat" % besteps

    labels = np.loadtxt(fn).flatten()
    xs = np.arange(1,traj.n_frames+1,1)
# Cluster membership by weight
    fig = plt.figure(figsize=(11,8.5))
    ax = fig.gca()
    for i in set(labels):
        i = int(i)
        if i == -1:
            ax.scatter(xs[labels == i], frameweights[-1][labels == i], c=colordict[i], label="Outliers")
        else:
            ax.scatter(xs[labels == i], frameweights[-1][labels == i], c=colordict[i], label="Cluster %d" % (i+1))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Relative weight")
    ax.legend(frameon=False)
    plt.savefig("Cluster_memberships_byweight_gamma_1e+03_eps_%.2e.png" % besteps, dpi=300)
