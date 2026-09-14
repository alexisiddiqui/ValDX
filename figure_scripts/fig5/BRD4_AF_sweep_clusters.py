# %%
### ValDXer testing
import os
os.environ["HDXER_PATH"] = "/home/alexi/Documents/HDXer"

import sys
sys.path.append("/home/alexi/Documents/ValDX/")

from ValDX.ValidationDX import ValDXer
from ValDX.VDX_Settings import Settings
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter

import numpy as np
settings = Settings(name='BRD4')
# settings.replicates = 1
settings.gamma_range = (1,8)
settings.train_frac = 0.5
settings.RW_exponent = [0]
settings.split_mode = 'R3'
# settings.stride = 1000
# settings.HDXer_stride = 10000

# settings.RW_do_reweighting = True
# settings.RW_do_params = False
import pickle

VDX = ValDXer(settings)
expt_name = 'Experimental'
test_name = "BRD4_af_small"

import cProfile
import pstats

# %%
import mdtraj as md

# %%
### add code to read in sequence from CIF file instead of copying it manually

cif_file = "raw_data/BRD4/BRD4_APO/AF-O60885-F1-model_v4.cif"

sequence_header = "_entity_poly.pdbx_seq_one_letter_code"
sequence = ""
seq_head_idx = 0
with open(cif_file, 'r') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if sequence_header in line:
            seq_head_idx = idx+1
            break
    
    for idx, line in enumerate(lines[seq_head_idx:]):
        if idx > 0 and line[0] == ";":
            break
        sequence += line.strip()


# print(sequence)



# strip sequence of non letters
sequence = ''.join([i for i in sequence if i.isalpha()])

print(sequence)

print("Sequence length: ", len(sequence))


# %%


# %% [markdown]
# 

# %%
# # convert sequence to FASTA format
# def write_fasta(sequence, header, file_name):
#     """
#     Writes a single-letter amino acid sequence to a FASTA file.
    
#     Parameters:
#     - sequence: A string containing the amino acid sequence.
#     - header: A string to be used as the header in the FASTA file.
#     - file_name: The name of the FASTA file to be created.
#     """
#     print(f"Writing sequence to {file_name}")
#     with open(file_name, 'w') as fasta_file:
#         # Write the header with the '>' symbol
#         fasta_file.write(f">{header}\n")
        
#         # Write the sequence in lines of 80 characters
#         for i in range(0, len(sequence), 80):
#             fasta_file.write(sequence[i:i+80] + "\n")


# %%
# # fasta_path = os.path.join("raw_data", "BRD4", 'BRD4_APO.fasta')
# write_fasta(sequence, 'LXRa', fasta_path)


# %%


# %%
# raw_hdx_path = "raw_data/BRD4/BRD4_APO/ELN55049_AllResultsTables_Curated.csv"
# raw_hdx = pd.read_csv(raw_hdx_path)
# raw_hdx.head()

# %%


# %%
# # convert FD in DeutTime to -1
# raw_hdx["Exposure"] = raw_hdx["DeutTime"].replace('FD', -1)

# # remove 's' from Deuteration Time
# raw_hdx["Exposure"] = raw_hdx["Exposure"].str.replace('s', '').astype(float)

# # replace NaN with -1
# raw_hdx["Exposure"].fillna(-1, inplace=True)

# raw_hdx.head()

# %%

# # print entire dataframe
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# print(raw_hdx.loc[raw_hdx["Exposure"] == 0]["Uptake"])

# %%
# print(raw_hdx.loc[raw_hdx["Exposure"] == 0].Uptake.value_counts(dropna=False))

# # fill NaNs with 0
# raw_hdx["Uptake"].fillna(0, inplace=True)

# %%
# # group by Start and End to extract peptide using ngroup
# raw_hdx["Peptide"] = raw_hdx.groupby(["Start", "End"]).ngroup()

# raw_hdx.head()



# %%
# # average Uptake for each peptide and Exposure
# hdx = raw_hdx.groupby(["Start","End","Peptide", "Exposure"])["Uptake"].mean().reset_index()

# print(hdx)

# %%
# # select Exposure -1
# max_uptake = hdx.loc[hdx["Exposure"] == -1]["Uptake"].values

# print(max_uptake)

# no_exposure_times = hdx["Exposure"].unique()
# print(len(no_exposure_times))

# # extend max_uptake to all Exposure times (each elemetn should be repeated len(no_exposure_times) times) ie [[m]*no_exposure_times for m in max_uptake]
# max_uptake = [m for m in max_uptake for _ in range(len(no_exposure_times))]
# print(max_uptake)


# # add max_uptake to hdx
# hdx["MaxUptake"] = max_uptake


# %%
# print(hdx)

# %%
# hdx['UptakeFraction'] = hdx['Uptake'] / hdx['MaxUptake']

# hdx.head()

# %%
# # remove Exposure -1
# hdx = hdx.loc[hdx["Exposure"] != -1]

# hdx.head()

# %%

# # pivot exposure and uptake fraction
# hdx = hdx.groupby(['Start', 'End', 'Exposure'])['UptakeFraction'].mean().reset_index()

# print(hdx)




# %%
# # clamp UptakeFraction to 1
# hdx["UptakeFraction"] = hdx["UptakeFraction"].clip(upper=1)
# print(hdx)


# %%
# # conver to HDXer format ie start, end, exposure_1, exposure_2 

# # pivot so that exposure time is the column name drop the exposure column
# hdx = hdx.pivot(index=['Start', 'End'], columns='Exposure', values='UptakeFraction').reset_index()

# # change Start to ResStr and End to ResEnd
# hdx = hdx.rename(columns={'Start': 'ResStr', 'End': 'ResEnd'})

# # drop the exposure column
# hdx.columns.name = None

# print(hdx)


# %%
# print(hdx)


# %%

# hdx = hdx.round(5)
# hdx.to_csv(os.path.join("raw_data", "BRD4", 'BRD4_APO.dat'), sep=' ', index=False)


# %%
# segs = raw_hdx[['Start', 'End']].drop_duplicates().sort_values(by=['Start', 'End']).reset_index(drop=True)


# %%

# # convert to list of tuples
# segs = [tuple(x) for x in segs.values]

# print(segs)


# %%


# # write list as new lines with space delimiter
# with open(os.path.join("raw_data", "BRD4", 'BRD4_APO_segs.txt'), 'w') as f:
#     for item in segs:
#         f.write("%s\n" % ' '.join(map(str, item)))

# %%
# from pdbfixer import PDBFixer
# from openmm.app import PDBFile

# BPTI_dir = "/Users/alexi/Library/CloudStorage/OneDrive-Nexus365/Rotation_Projects/Rotation_3/Project/ValDX/raw_data/BRD4/BRD4_APO"
# sim_dir = os.path.join(BPTI_dir, "alphafold_quick")

# pdb_list = [f for f in os.listdir(sim_dir) if f.endswith('.pdb')]

# print(pdb_list) 


# H_sim_dir = os.path.join(BPTI_dir, "alphafold_H")

# os.makedirs(H_sim_dir, exist_ok=True)

# for pdb in pdb_list:
#     fixer = PDBFixer(os.path.join(sim_dir, pdb))
#     fixer.addMissingHydrogens(7.0)
#     H_pdb_name = pdb.replace('.pdb', '_H.pdb')
#     PDBFile.writeFile(fixer.topology, fixer.positions, open(os.path.join(H_sim_dir, H_pdb_name), 'w'), keepIds=True)

# pdb_list = [f for f in os.listdir(H_sim_dir) if f.endswith('.pdb')]



# %% [markdown]
# Generate conformations with Alphafold
# 
# # need to find out how to generate a wide range of conformations

# %%
def pre_process_main():
    # BPTI data
    BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/BRD4/BRD4_APO"

    # BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI"

    os.listdir(BPTI_dir)

    segs_name = "BRD4_APO_segs.txt"
    segs_path = os.path.join(BPTI_dir, segs_name)

    hdx_name = "BRD4_APO_clean.dat"
    hdx_path = os.path.join(BPTI_dir, hdx_name)
    print(hdx_path)

    rates_name = "out__train_MD_Simulated_1Intrinsic_rates.dat"
    rates_path = os.path.join(BPTI_dir, rates_name)
    sim_name = 'BRD4_AF'

    sim_dir = os.path.join(BPTI_dir, "alphafold_quick")

    pdb_list = [f for f in os.listdir(sim_dir) if f.endswith('.pdb')]

    print(pdb_list) 


    H_sim_dir = os.path.join(BPTI_dir, "alphafold_H")

    os.makedirs(H_sim_dir, exist_ok=True)

    for pdb in pdb_list:
        continue
        fixer = PDBFixer(os.path.join(sim_dir, pdb))
        fixer.addMissingHydrogens(7.0)
        H_pdb_name = pdb.replace('.pdb', '_H.pdb')
        PDBFile.writeFile(fixer.topology, fixer.positions, open(os.path.join(H_sim_dir, H_pdb_name), 'w'), keepIds=True)

    pdb_list = [f for f in os.listdir(H_sim_dir) if f.endswith('.pdb')]
    print(pdb_list)

    top_path = "/home/alexi/Documents/ValDX/raw_data/BRD4/BRD4_APO/BRD4_APO_484_1_af_sample_127_10000_protonated.pdb"
    # pdb_paths = [os.path.join(H_sim_dir, i) for i in pdb_list]

    # print("top",top_path)


    # print(pdb_paths)

    # small_traj_path = top_path.replace(".pdb","_small.xtc")
    # # small_traj_path = os.path.join(sim_dir, small_traj_name)

    # u = mda.Universe(top_path, pdb_paths)

    # print(small_traj_path)
        
    # with XTCWriter(small_traj_path, n_atoms=u.atoms.n_atoms) as W:
    #     for ts in u.trajectory:
    #         W.write(u.atoms)

    # traj_paths = [os.path.join(sim_dir, i) for i in os.listdir(sim_dir) if i.endswith(".pdb")]
    
    traj_paths = ["/home/alexi/Documents/ValDX/raw_data/BRD4/BRD4_APO/BRD4_APO_484_1_af_sample_127_10000_protonated.xtc"]
    # print(traj_paths)
    # u = mda.Universe(top_path, *traj_paths)

    # small_traj_name = top_path.replace(".pdb","_small.xtc")
    # small_traj_path = os.path.join(sim_dir, small_traj_name)

    # with XTCWriter(small_traj_path, n_atoms=u.atoms.n_atoms) as W:
    #     for ts in u.trajectory[1:2]:
    #         W.write(u.atoms)
    #         W.write(u.atoms)
    #         break
    # print(traj_paths)
    # traj_paths = [small_traj_path]

    return hdx_path, segs_path, rates_path, top_path, traj_paths, sim_name, expt_name, test_name


# %%
hdx_path, segs_path, rates_path, top_path, traj_paths, sim_name, expt_name, test_name = pre_process_main()

# %%
# combined_analysis_dump, names, save_paths = VDX.run_benchmark_ensemble(system=test_name,
#                                                                         times=[0.0, 15.0, 60.0, 600.0, 3600.0, 14400.0],
#                                                                         expt_name=expt_name,
#                                                                         n_reps=1,
#                                                                         split_modes=['r','s','R3'],
#                                                                         RW=True,
#                                                                         hdx_path=hdx_path,
#                                                                         segs_path=segs_path,
#                                                                         traj_paths=traj_paths,
#                                                                         top_path=top_path)

                   

times = [0.0, 15.0, 60.0, 600.0, 3600.0, 14400.0]
times = [0.25,	1.0,	10.0,	60.0]

# combined_analysis_dump, names, save_paths = VDX.run_benchmark_ensemble(system=test_name,
#                                                                         times=times,
#                                                                         expt_name=expt_name,
#                                                                         n_reps=4,

#                                                                         optimise=False,
#                                                                         hdx_path=hdx_path,
#                                                                         segs_path=segs_path,
#                                                                         traj_paths=traj_paths,
#                                                                         top_path=top_path)

                                                                        

# # run BV optimisation

# combined_analysis_dump, names, save_paths = VDX.run_benchmark_ensemble(system=test_name,
#                                                                         times=times,
#                                                                         expt_name=expt_name,
#                                                                         n_reps=4,
#                                                                         RW=True,
#                                                                         optimise=True,
#                                                                         hdx_path=hdx_path,
#                                                                         segs_path=segs_path,
#                                                                         traj_paths=traj_paths,
#                                                                         top_path=top_path)
VDX.run_sweep_cluster2_ensemble(system=test_name,
                                times=times,
                                expt_name=expt_name,
                                n_reps=3,
                                # split_modes=['R3'],
                                hdx_path=hdx_path,
                                segs_path=segs_path,
                                traj_paths=traj_paths,
                                top_path=top_path)