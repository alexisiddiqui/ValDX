#!/usr/bin/env python

# Script to insert atom-based b-factors into PDB with MDtraj
# (or equally to write out atom/residue b-factors to file)

import mdtraj as md
import numpy as np
import os
times = [0.167, 1.0, 10.0, 60.0, 120.0]
for idx, time in enumerate(times):

    # Load in PDB and desired residue segments
    inpdb = md.load(os.path.expandvars("ref_closed_state.pdb"))
    df_segs_diffs = np.loadtxt("closed-open_diffs.dat", dtype=[ ('res', np.int32, (1,)),\
                               ('fracs', np.float64, (5,))])

    #Atom based bfacs = list of lists
    bfacs = []
    for i in range(inpdb.n_atoms):
        bfacs.append([])
    for atm, dfracs in enumerate(bfacs):
        for line, seg in enumerate(df_segs_diffs['res'], start=0):
            if inpdb.topology.atom(atm).residue.resSeq == seg[0]:
                dfracs.append(df_segs_diffs['fracs'][line, idx]) # idx = time
            else:
                continue
    # convert to numpy, averaging all the lists
    # This will throw up a warning for nans
    bfac_array = np.asarray([ np.mean(np.asarray(x)) for x in bfacs ])
    bfac_array = np.nan_to_num(bfac_array)

    # Residue based bfacs
    resbfacs = []
    for i in range(inpdb.n_residues):
        resbfacs.append([])
    for res, dfracs in enumerate(resbfacs):
        for line, seg in enumerate(df_segs_diffs['res']):
            if res+1 == seg[0]:
                dfracs.append(df_segs_diffs['fracs'][line, idx]) # idx = time
            else:
                continue
    resbfac_array = np.asarray([ np.mean(np.asarray(x)) for x in resbfacs ])
    resbfac_array = np.nan_to_num(resbfac_array)


    np.savetxt("closed-open_full_%smin.txt" % time, bfac_array)
    np.savetxt("closed-open_full_byres_%smin.txt" % time, resbfac_array)
    inpdb.save_pdb("ref_closed_closed-open_full_%smin.pdb" % time, bfactors=bfac_array)
