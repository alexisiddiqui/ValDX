#!/usr/bin/env python

# Script to calculate cluster membership

import mdtraj as md
import numpy as np
import sys, os

# Import trajectory & calculate CA pairwise dists

trjdir = "../../data/trajectories"
top = md.load_topology(os.path.join(trjdir, "TeaA_ref_closed_state.pdb"))
traj = md.load(os.path.join(trjdir, "TeaA_initial_reimaged.xtc"), top=top)

# Choose epsilon to be an average of +/- cutoff A from every point
cutoffs = np.linspace(0.05, 0.5, 10)
epsilons = np.sqrt((cutoffs**2)*traj.n_frames)

weightsdir = "../../figures/fig3/data_folder"
weightsfile = os.path.join(weightsdir, "mixed_gamma_1x10^3_final_weights.dat")
frameweights = np.loadtxt(weightsfile)*traj.n_frames



for eps in epsilons:
    fn = "Gamma_1e+03_cluster_labels_eps_%.2e.dat" % eps
    labels = np.loadtxt(fn).flatten()
    f = open("Gamma_1e+03_eps_%.2e_populations.dat" % eps, 'a')
    f.write("# Cluster Population\n")
    for i in set(labels):
        i = int(i)
        popn = np.sum(frameweights[labels == i]) / np.sum(frameweights)
        f.write("%d %8.6f\n" % (i, popn))
    
f.close() 

# Choose the best epsilon (ave RMSD = 0.20?) & save trajectory
besteps = epsilons[3]
fn = "Gamma_1e+03_cluster_labels_eps_%.2e.dat" % besteps
labels = np.loadtxt(fn).flatten()
traj[0].save_pdb("Clustering_topology.pdb")
for i in set(labels):
    i = int(i)
    traj[labels == i].save_dcd("Cluster_%d_eps_%.2e.dcd" % (i, besteps))


