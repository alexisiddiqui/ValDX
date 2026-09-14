#!/usr/bin/env python

# Script to sum segment averages and take diffs Out - In 

import numpy as np
import os


# Setup variables
times = [0.167, 1., 10., 60., 120.]
folder1 = os.path.expandvars("./data_folder/closed")
folder2 = os.path.expandvars("./data_folder/open")
labels = ("closed", "open")
runs = 1


# Import files
results1 = []
for run in range(1, runs+1):
    results1.append(np.loadtxt(os.path.join(folder1, "closed_SUMMARY_residue_fractions.dat"), dtype=[('res', np.int32, (1,)), ('fracs', np.float64, (len(times),))]))


results2 = []
for run in range(1, runs+1):
    results2.append(np.loadtxt(os.path.join(folder2, "open_SUMMARY_residue_fractions.dat"), dtype=[('res', np.int32, (1,)), ('fracs', np.float64, (len(times),))]))

print("Comp segs equal?", np.array_equal(results1[0]['res'], results2[0]['res']))

mean1 = np.mean([_['fracs'] for _ in results1], axis=0)
mean2 = np.mean([_['fracs'] for _ in results2], axis=0)

diffs = mean1 - mean2

# The "%s-%s_diffs.dat" file contains the residue-based deuterated
# fraction differences and residue numbers and can be used to recreate
# the plot without the links to the original data.
np.savetxt("%s-%s_diffs.dat" % labels, np.concatenate((results1[0]['res'], diffs), axis=1), \
      fmt="%3d " + "%8.5f "*5, header="ResID  Times / min: 0.167 1.0 10.0 60.0 120.0")
