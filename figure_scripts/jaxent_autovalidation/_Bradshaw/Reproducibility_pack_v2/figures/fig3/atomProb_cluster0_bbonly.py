# For use with pyXplor or xplor -py only 
# (so that python env is already set up)
# 
# Script to read in multiple PDB files and
# perform atom probability density map

from xplor import command
from pdbTool import PDBTool
from atomProb import AtomProb
from atomSel import AtomSel

sim = xplor.simulation
command("set echo off messages off end")
command("structure @ref_closed_state_prot.psf end") # might not be necessary?

num_frames = 31133
structures = [ "Cluster_0_eps_4.22e+01_bbonly.pdb.%d" % i for i in range(1,num_frames+1) ]

atom_sel_text = "name c or name ca or name n"

atom_positions = []
# We don't do fitting, we just read in coords. Structures must be pre-fit
for struct in structures:
    pdbobj = PDBTool(struct, AtomSel(atom_sel_text))
    pdbobj.read()
    atom_positions.append(sim.atomPosArr())

print("Frames read in: %d" % len(atom_positions))
print("Last 20 coords of first frame: %s" % atom_positions[0][-20:])
print("Last 20 coords of last frame: %s" % atom_positions[-1][-20:])
print("(These may include some unknown coords, but as long as your atom selection is ok, this is fine!)")

# Create density map object
probmap = AtomProb( AtomSel(atom_sel_text), atom_positions)
# Grid dimensions & buffer outside molecule
probmap.gridVals.xdelta=0.25
probmap.gridVals.ydelta=0.25
probmap.gridVals.zdelta=0.25
probmap.gridVals.cushion=2.5
# set default atomic radius
#probmap.setAtomRadius(1.4)

# Descriptions from https://nmr.cit.nih.gov/xplor-nih/doc/current/python/ref/atomProb.html
# specify that no normalization be applied:
# for each atom contribution in the
# supplied atomPosList array a normalized
# distribution will be contributed to the
# grid
probmap.setScaleType("flat")
probmap.calc()
probmap.writeEDM("BBonly_cluster_0_script_flat.xplor")
# contribution from each atom will be
# normalized by the maximum contribution
# from that atom
probmap.setScaleType("amplitude")
probmap.calc()
probmap.writeEDM("BBonly_cluster_0_script_amplitude.xplor")
# same as amplitude, but the prefactor
# is divided by an additional
# factor proportional to volume
probmap.setScaleType("normalize")
probmap.calc()
probmap.writeEDM("BBonly_cluster_0_script_normalize.xplor")

