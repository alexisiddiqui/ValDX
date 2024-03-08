# %%
### ValDXer testing
import os
os.environ["HDXER_PATH"] = "/home/alexi/Documents/HDXer"

from ValDX.ValidationDX import ValDXer
from ValDX.VDX_Settings import Settings
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter

from pdbfixer import PDBFixer
from openmm.app import PDBFile

settings = Settings(name='testHOIP')
settings.replicates = 1
settings.gamma_range = (2,6)
settings.train_frac = 0.5
settings.RW_exponent = [0]
settings.split_mode = 'R3'
settings.stride = 1000
# settings.HDXer_stride = 10000

settings.RW_do_reweighting = True
settings.RW_do_params = False
import pickle

VDX = ValDXer(settings)
expt_name = 'Experimental'
test_name = "HOIPapo_test"
import icecream as ic
# ic.disable()

# %% [markdown]
# 

# %%
# ### add code to read in sequence from CIF file instead of copying it manually

# cif_file = "/Users/alexi/Library/CloudStorage/OneDrive-Nexus365/Rotation_Projects/Rotation_3/Project/ValDX/raw_data/HOIP/HOIP_apo/AF-Q96EP0-F1-model_v4.cif"

# sequence_header = "_entity_poly.pdbx_seq_one_letter_code"
# sequence = ""
# seq_head_idx = 0
# with open(cif_file, 'r') as f:
#     lines = f.readlines()
#     for idx, line in enumerate(lines):
#         if sequence_header in line:
#             seq_head_idx = idx+1
#             break
    
#     for idx, line in enumerate(lines[seq_head_idx:]):
#         if idx > 0 and line[0] == ";":
#             break
#         sequence += line.strip()


# # print(sequence)



# # strip sequence of non letters
# sequence = ''.join([i for i in sequence if i.isalpha()])

# print(sequence)

# print("Sequence length: ", len(sequence))


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

# fasta_path = os.path.join("raw_data", "HOIP", 'HOIP_apo.fasta')
# write_fasta(sequence, 'HOIPapo', fasta_path)



# %%


# %% [markdown]
# 

# %%


# %%


# %%
raw_hdx_path = "/home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo/HOIP_apo_peptide.csv"
raw_hdx = pd.read_csv(raw_hdx_path)
raw_hdx.tail()

# %%
# drop Unnamed: 0	

raw_hdx = raw_hdx.drop(columns=['Unnamed: 0'])
raw_hdx.head()


# %%
# assign peptide number for each start and end residue using ngroup
raw_hdx['peptide'] = raw_hdx.groupby(['Start','End']).ngroup()

raw_hdx.head()

# %%

times = [0, 0.5, 5.0]

num_peptides = len(raw_hdx)//len(times)

exposure = times * num_peptides

raw_hdx['Exposure'] = exposure

raw_hdx.head()

# %%
raw_hdx['UptakeFraction'] = raw_hdx['Uptake'] / raw_hdx['MaxUptake']

raw_hdx.head()

# %%
# clamp UptakeFraction to 1
raw_hdx['UptakeFraction'] = raw_hdx['UptakeFraction'].clip(upper=1)

# %%
# # print entire dataframe
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# print(raw_hdx)



# %%


# %%


# %%

# pivot exposure and uptake fraction
grouped = raw_hdx.pivot(index=['Start', 'End'], columns='Exposure', values='UptakeFraction').reset_index()

# drop 
grouped.head()


# %%

# print entire dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(grouped)

# %%
# conver to HDXer format ie start, end, exposure_1, exposure_2 

# change Start to ResStr and End to ResEnd
hdx = grouped.rename(columns={'Start': 'ResStr', 'End': 'ResEnd'})

# drop the exposure column
hdx.columns.name = None

print(hdx)


# %%


# %%

hdx = hdx.round(5)
hdx.to_csv(os.path.join("raw_data", "HOIP", 'HOIP_apo.dat'), sep=' ', index=False)


# %%
segs = hdx[['ResStr', 'ResEnd']].drop_duplicates().sort_values(by=['ResStr', 'ResEnd']).reset_index(drop=True)


# %%

# # convert to list of tuples
# segs = [tuple(x) for x in segs.values]

# print(segs)


# %%


# # write list as new lines with space delimiter
# with open(os.path.join("raw_data", "HOIP", 'HOIP_APO_segs.txt'), 'w') as f:
#     for item in segs:
#         f.write("%s\n" % ' '.join(map(str, item)))

# %%
# ### at the moment PDB fixer is adding different number of hydrogens to different structures... Need to change the code to use PROPKA to get H states and apply to all strucutres
# BPTI_dir = "/Users/alexi/Library/CloudStorage/OneDrive-Nexus365/Rotation_Projects/Rotation_3/Project/ValDX/raw_data/HOIP/HOIP_apo/"
# sim_dir = os.path.join(BPTI_dir, "alphafold_quick")

# pdb_list = [f for f in os.listdir(sim_dir) if f.endswith('.pdb')]

# print(pdb_list) 


# H_sim_dir = os.path.join(BPTI_dir, "alphafold_H")

# os.makedirs(H_sim_dir, exist_ok=True)

# for pdb in pdb_list:
#     continue
#     fixer = PDBFixer(os.path.join(sim_dir, pdb))
#     fixer.addMissingHydrogens(7.0)
#     H_pdb_name = pdb.replace('.pdb', '_H.pdb')
#     PDBFile.writeFile(fixer.topology, fixer.positions, open(os.path.join(H_sim_dir, H_pdb_name), 'w'), keepIds=True)

# pdb_list = [f for f in os.listdir(H_sim_dir) if f.endswith('.pdb')]



# top_path = os.path.join(H_sim_dir, pdb_list[0])
# pdb_paths = [os.path.join(H_sim_dir, i) for i in pdb_list]

# print(top_path)
# print(pdb_paths)


# small_traj_name = top_path.replace(".pdb","_small.xtc")
# small_traj_path = os.path.join(H_sim_dir, small_traj_name)

# u = mda.Universe(top_path)
    
# with XTCWriter(small_traj_path, n_atoms=u.atoms.n_atoms) as W:
#     for ts in u.trajectory:
#         W.write(u.atoms)
#         W.write(u.atoms)
#         break


# %% [markdown]
# Generate conformations with Alphafold
# 
# # need to find out how to generate a wide range of conformations

# %%
def pre_process_main():
    # BPTI data
    # BPTI_dir = "/Users/alexi/Library/CloudStorage/OneDrive-Nexus365/Rotation_Projects/Rotation_3/Project/ValDX/raw_data/HOIP/HOIP_apo/"
    BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo"
    # BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI"

    sim_name = 'HOIP_apo_AF'
    os.listdir(BPTI_dir)

    segs_name = "HOIP_APO_segs.txt"
    segs_path = os.path.join(BPTI_dir, segs_name)

    hdx_name = "HOIP_apo.dat"
    hdx_path = os.path.join(BPTI_dir, hdx_name)
    print(hdx_path)

    rates_name = "out__train_MD_Simulated_1Intrinsic_rates.dat"
    rates_path = os.path.join(BPTI_dir, rates_name)

    sim_dir = os.path.join(BPTI_dir, "alphafold_quick")

    pdb_list = [f for f in os.listdir(sim_dir) if f.endswith('.pdb')]

    print(pdb_list) 


    H_sim_dir = os.path.join(BPTI_dir, "alphafold_H")

    os.makedirs(H_sim_dir, exist_ok=True)

    for pdb in pdb_list:
        continue
        fixer = PDBFixer(os.path.join(H_sim_dir, pdb))
        fixer.addMissingHydrogens(7.0)
        H_pdb_name = pdb.replace('.pdb', '_H.pdb')
        PDBFile.writeFile(fixer.topology, fixer.positions, open(os.path.join(H_sim_dir, H_pdb_name), 'w'), keepIds=True)
        break
    pdb_list = [f for f in os.listdir(H_sim_dir) if f.endswith('.pdb')]


    top_path = "/home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo/HOIP_apo697_af_sample_1000_protonated.pdb"
    # pdb_paths = [os.path.join(H_sim_dir, i) for i in pdb_list]

    # print(top_path)
    # print(pdb_paths)


    # small_traj_name = top_path.replace(".pdb","_small.xtc")
    # small_traj_path = os.path.join(sim_dir, small_traj_name)

    # u = mda.Universe(top_path)
        
    # with XTCWriter(small_traj_path, n_atoms=u.atoms.n_atoms) as W:
    #     for ts in u.trajectory:
    #         W.write(u.atoms)
    #         W.write(u.atoms)
    #         break
    # # traj_paths = [os.path.join(sim_dir, i) for i in os.listdir(sim_dir) if i.endswith(".pdb")]
    
    traj_paths = ["/home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo/HOIP_apo697_af_sample_1000_protonated.xtc"]

    print(traj_paths)
    return hdx_path, segs_path, rates_path, top_path, traj_paths, sim_name, expt_name, test_name


# %%
hdx_path, segs_path, rates_path, top_path, traj_paths, sim_name, expt_name, test_name = pre_process_main()

# %%
# # ic.disable()
# combined_analysis_dump, names, save_paths = VDX.run_benchmark_ensemble(system=test_name,
#                                                                     times=[0, 0.5, 5.0],
#                                                                     expt_name=expt_name,
#                                                                     n_reps=1,
#                                                                     optimise=False,
#                                                                     # split_modes=['r'],
#                                                                     hdx_path=hdx_path,
#                                                                     segs_path=segs_path,
#                                                                     traj_paths=traj_paths,
#                                                                     top_path=top_path)

settings.cluster_frac1 = 0.1
combined_analysis_dump, names, save_paths = VDX.run_refine_ensemble(system=test_name+"_modal",
                                                                    times=[0, 0.5, 5.0],
                                                                    expt_name=expt_name,
                                                                    n_reps=2,
                                                                    split_mode='R3',
                                                                    hdx_path=hdx_path,
                                                                    segs_path=segs_path,
                                                                    traj_paths=traj_paths,
                                                                    top_path=top_path,
                                                                    modal_cluster=True)

# %%

combined_analysis_dump, names, save_paths = VDX.run_refine_ensemble(system=test_name+"_mean",
                                                                    times=[0, 0.5, 5.0],
                                                                    expt_name=expt_name,
                                                                    n_reps=2,
                                                                    split_mode='R3',
                                                                    hdx_path=hdx_path,
                                                                    segs_path=segs_path,
                                                                    traj_paths=traj_paths,
                                                                    top_path=top_path,
                                                                    modal_cluster=False)
