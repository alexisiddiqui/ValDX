#!/usr/bin/env python

# Script to take list of input trajectories and create artificial by-residue DFs for target data

import numpy as np
import sys, os, argparse
from glob import glob
from copy import deepcopy

### Argparser ###
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folders", \
                        help="Folder(s) in which to find contacts/Hbond files for analysis", \
                        nargs='+', type=str, required=True)
    parser.add_argument("-w","--weights", \
                        help="List of relative weights for subsets of the total trajectory. Weights should sum to 1, or they will be normalized to sum to 1. E.g. '-w 0.6 0.4' will split the trajectory into two parts, weighted 60/40. No default.", \
                        nargs='+', type=float, required=True)
    parser.add_argument("-n","--numframes", \
                        help="Number of frames in each subset of the total trajectory. Length should equal the number of subsets, and the sum should equal the total number of frames. E.g. '-n 2000 1000' will split the trajectory into two parts, the first of 2000 frames, the second of 1000. No default", \
                        nargs='+', type=int, required=True)
    parser.add_argument("-r","--ratefile", \
                        help="Path of file with intrinsic rates. These should be defined as '$resid $rate', whitespace separated, with one line pre residue. If the input trajectories do not have the same number of residues, the trajectories will be fitted based on identified common residues only. No default.", \
                        required=True)
    parser.add_argument("-exp","--expfile", \
                        help="Path of file with experimental deuterated fractions. Fractions will be calculated for the segments in this file. Segs should be defined one per line, followed by one column for each timepoint in --times, Whitespace separated. No default.", \
                        required=True)
    parser.add_argument("-dt","--times", \
                        help="Times for analysis, in minutes. Defaults to [ 0.167, 1.0, 10.0, 120.0 ]", \
                        nargs='+', default=[0.167, 1.0, 10.0, 120.0], type=float)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

args = parse()
folderlist = args.folders
framelist = args.numframes
weightlist = args.weights
expt_path = args.expfile
kintfile = args.ratefile
times = args.times


# Define funcs to read in contacts & H-bonds
def strip_filename(fn, extrastr=""):
    """Sorting key that will strip the integer residue number
       from a Contacts/Hbonds filename. Expects filenames of 
       the sort 'Contacts_123.tmp' - splits on _ and . 

       If filenames are not quite of this format, optionally
       the 'extrastr' argument can be used to add an additional
       string to the filename, e.g. 'Contacts_chain_0_res_123.tmp'"""
    try:
        _ = fn.split("Contacts_"+extrastr)[1]
        return int(_.split(".")[0])
    except:
        pass
    try:
        _ = fn.split("Hbonds_"+extrastr)[1]
        return int(_.split(".")[0])
    except:
        raise NameError("File found without correct 'Contacts_' or 'Hbonds_' format filename: %s" % fn)

# Read in single column datafiles        
def files_to_array(fnames):
    """Read in data fom list of files with np.loadtxt. 
       Returns array of shape (n_files, n_data_per_file)"""
    l = [ np.loadtxt(f) for f in fnames ]
    try:
        return np.stack(l, axis=0)
    except ValueError:
        raise ValueError("Error in stacking files read with np.loadtxt - are they all the same length?")

# Read in initial contacts & H-bonds.
# Store as 2D-np arrays of shape (n_residues, n_frames)

# Treat separate chains as separate frames. Can be upweighted/downweighted individually
# (Only applicable if we add extra lines here to read in the extra Contacts/Hbonds files for each chain)
contactfiles, hbondfiles = [],[]
for folder in folderlist:
    contactfiles.append(sorted(glob(os.path.join(folder, "Contacts_chain_0*.tmp")), key=lambda x: strip_filename(x, extrastr="chain_0_res_")))
    hbondfiles.append(sorted(glob(os.path.join(folder, "Hbonds_chain_0*.tmp")), key=lambda x: strip_filename(x, extrastr="chain_0_res_")))

resids = []
# This is a list comprehension with the try/except for the extra strings
for curr in contactfiles:
    _ = []
    for f in curr:
        try:
            _.append( strip_filename(f, extrastr="chain_0_res_") )
        except NameError:
            _.append( strip_filename(f, extrastr="chain_1_res_") ) # E.g. for 2 chain system
    resids.append(_)

sorted_resids = deepcopy(resids)
sorted_resids.sort(key=lambda _: len(_))
filters = list(map(lambda _: np.in1d(_, sorted_resids[0]), resids)) # Get indices to filter by shortest
new_resids = []
for r, f in list(zip(resids, filters)):
    new_resids.append(np.array(r)[f])
new_resids = np.stack(new_resids)
if not np.diff(new_resids, axis=0).sum(): # If sum of differences between filtered resids == 0
    pass
else:
    raise ValueError("Error in filtering trajectories to common residues - do residue IDs match up in your intrinsic rate files?")

_contacts = list(map(lambda x, y: x[y], [ files_to_array(curr_cfiles) for curr_cfiles in contactfiles ], filters))
_hbonds = list(map(lambda x, y: x[y], [ files_to_array(curr_hfiles) for curr_hfiles in hbondfiles ], filters))


contacts = np.concatenate(_contacts, axis=1)
print("Contacts read")
hbonds = np.concatenate(_hbonds, axis=1)
print("Hbonds read")

# Some basic asserts
assert(len(contacts) == len(hbonds))
assert(sum(framelist) == len(contacts[0]))
assert(len(weightlist) == len(framelist))
# Normalize weights & prepend framelist with initial index
weightlist = np.array(weightlist)
weightlist /= sum(weightlist)
framelist.insert(0,0)
framelist = np.array(framelist)

endframes = np.cumsum(framelist)
for j, weight in enumerate(weightlist, 1):
    # Length of weightlist should be 1 shorter than framelist, so index i shouldn't overrun
    i = j-1
    try:
        contacts[:,endframes[i]:endframes[j]] *= weight/framelist[j]
        hbonds[:,endframes[i]:endframes[j]] *= weight/framelist[j]
    except IndexError:
        print("Index overran, something might be wrong!")
        contacts[:,endframes[i]:] *= weight/framelist[j]
        hbonds[:,endframes[i]:] *= weight/framelist[j]
    
ave_contacts = np.sum(contacts, axis=1)
ave_hbonds = np.sum(hbonds, axis=1)

# Read intrinsic rates, multiply by times
kint = np.loadtxt(kintfile, usecols=(1,)) # We only need one file here and it'll be filtered based on its residue IDs
kintresid = np.loadtxt(kintfile, usecols=(0,))
kintfilter = np.in1d(kintresid, sorted_resids[0])
kint = kint[kintfilter]
resid = kintresid[kintfilter]
kint = np.repeat(kint[:, np.newaxis], len(times), axis=1)*times # Make sure len(times) is no. of expt times
# Read deuterated fractions, shape will be (n_residues, n_times)
exp_dfrac = np.loadtxt(expt_path, usecols=tuple(range(2,2+len(times))))
segments = np.loadtxt(expt_path, usecols=(0,1), dtype=np.int32)

# convert expt to (segments, residues, times)
exp_dfrac = exp_dfrac[:,np.newaxis,:].repeat(len(hbonds), axis=1)
# convert kint to (segments, residues, times)
kint = kint[np.newaxis,:,:].repeat(len(segments), axis=0)

# Make a set of filters that defines the residues in each segment & timepoint
segfilters=[]
for seg in segments:
    seg_resids = range(seg[0], seg[1]+1)
    segfilters.append(np.in1d(resid, seg_resids[1:])) # Filter but skipping first residue in segment
segfilters = np.array(segfilters)
segfilters = np.repeat(segfilters[:, :, np.newaxis], len(times), axis=2) # Repeat to shape (n_segments, n_residues, n_times)

assert all((segfilters.shape == exp_dfrac.shape, \
           segfilters.shape == kint.shape, \
           contacts.shape == hbonds.shape))

print("Segments and experimental dfracs read")

normseg = np.sum(segfilters)
with open("all_chisquareds.dat", 'w') as f:
    f.write("chisquare Bh Bc\n")

# Calculate Dfs
for Bh in np.arange(1.0, 13.1, 0.1):
    for Bc in np.arange(0.10, 0.51, 0.01):
        avelnpi=np.sum(Bc*contacts + Bh*hbonds, axis=1)
        num = -kint*segfilters
        avelnpi = np.repeat(avelnpi[:,np.newaxis], len(times), axis=1)
        avelnpi = avelnpi[np.newaxis,:,:].repeat(len(segments), axis=0)
        denom = avelnpi*segfilters
        byres_deutfrac = 1.0 - np.exp(np.divide(num, np.exp(denom), out=np.full(num.shape, np.nan), where=denom!=0))
        byseg_deutfrac = np.nanmean(byres_deutfrac, axis=1)
        #  Save
        np.savetxt("mixed_60-40_BH_%2.1f_BC_%3.2f_deuterated_fracs.dat" % (Bh, Bc), byseg_deutfrac, fmt="%8.5f")
        # Equivalent to reweighting script:
        byseg_deutfrac = byseg_deutfrac[:,np.newaxis,:].repeat(segfilters.shape[1], axis=1)
        chisquare = np.sum((byseg_deutfrac*segfilters - exp_dfrac*segfilters)**2) / normseg
        with open("all_chisquareds.dat", 'a') as f:
            f.write("%14.12f %3.2f %3.2f\n" % (chisquare, Bh, Bc))
        print("Done BH=%2.1f BC=%3.2f" % (Bh, Bc))

#        # Make segfracs
#        segsize = 10
#        residuefile = "mixed_60-40_artificial_expt_resfracs.dat"
#
#        startres = range(1,311, segsize-1)
#        endres = range(segsize,311, segsize-1) 
#
#        startres.pop(-1)
#        endres.pop(-1)
#        endres.append(310)
#
#        # For a seg aves file with byres information, use column 1, not 0, as the first residue of each segment is skipped
#        dfrac_byres = np.loadtxt("mixed_60-40_artificial_expt_resfracs.dat", usecols=(1,2,3,4,5,6))
#        # For a residue file use column 0, as the first label is the correct residue
#
#        with open("mixed_60-40_artificial_expt_segfracs10.dat", 'ab') as f:
#            f.write("# Res1 Res2, Time = 0.167, 1.0, 10.0, 60.0, 120.0 / min\n")
#        for segstart, segend in zip(startres, endres):
#            segdfracs = np.zeros(5) # Shape = no. of times
#            nres = 0
#            for r in range(segstart+1, segend+1): # Skip first residue
#                try:
#                    segdfracs += dfrac_byres[np.where(dfrac_byres[:,0] == r)][0,1:] # Final slice is just to extract dfracs from 2D array
#                    nres += 1
#                except IndexError: # e.g. empty array for prolines
#                    continue
#            segdfracs /= nres # Take mean
#            segs = np.array([segstart, segend])
#            _ = np.concatenate((segs, segdfracs))
#            np.savetxt(f, _[np.newaxis,:], fmt="%3d %3d %8.5f %8.5f %8.5f %8.5f %8.5f")



