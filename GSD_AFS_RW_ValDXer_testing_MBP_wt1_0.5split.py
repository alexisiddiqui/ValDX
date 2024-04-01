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
settings = Settings(name='test_MBP')
settings.replicates = 1
settings.gamma_range = (2,6)
settings.train_frac = 0.5
settings.RW_exponent = [0]
settings.split_mode = 'R3'
settings.stride = 1000
# settings.HDXer_stride = 10000

settings.RW_do_reweighting = False
settings.RW_do_params = True
import pickle

VDX = ValDXer(settings)
expt_name = 'Experimental'
test_name = "MBPwt1_test"

from icecream import ic

ic.disable()

# %%


# %%
# ### add code to read in sequence from CIF file instead of copying it manually

# cif_file = "/Users/alexi/Library/CloudStorage/OneDrive-Nexus365/Rotation_Projects/Rotation_3/Project/ValDX/raw_data/MBP/MaltoseBindingProtein/AF-P0AEX9-F1-model_v4.cif"

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

# fasta_path = os.path.join("raw_data", "MBP", 'MBP_wt.fasta')
# write_fasta(sequence, 'MBP_wt', fasta_path)



# %%


# %% [markdown]
# 

# %%


# %%


# %%
# raw_hdx_path = "/Users/alexi/Library/CloudStorage/OneDrive-Nexus365/Rotation_Projects/Rotation_3/Project/ValDX/raw_data/MBP/MaltoseBindingProtein/MBP analysis final editing export 2020 10 05_tidy.csv"
# raw_hdx = pd.read_csv(raw_hdx_path)
# raw_hdx.tail()

# %%
# state = "WT Null"

# raw_hdx = raw_hdx[raw_hdx['hx_sample'] == state]

# # drop nans in column d
# raw_hdx = raw_hdx.dropna(subset=['d'])

# raw_hdx.head()


# %%
# # group by pep_start and pep_end and hx_time and take the mean of the d values
# grouped = raw_hdx.groupby(['pep_start', 'pep_end',"hx_time"])["d"].mean().reset_index()

# grouped.head()

# # assign peptide number to each combination of pep_start and pep_end
# grouped['peptide']= grouped.groupby(['pep_start', 'pep_end']).ngroup()



# %%
# # print entire dataframe
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)

# print(grouped)


# %%
# # add MaxUptake column for each peptide

# # first, get the max uptake for each peptide
# max_uptake = grouped.groupby('peptide')['d'].max().reset_index()

# # assign the max uptake to each peptide
# grouped = grouped.merge(max_uptake, on='peptide', suffixes=('', '_max'))

# grouped.head()

# %%
# grouped['UptakeFraction'] = grouped['d'] / grouped['d_max']



# columns_to_drop = ['d', 'd_max']
# grouped = grouped.drop(columns=columns_to_drop)

# grouped.head()

# %%

# # pivot exposure and uptake fraction
# grouped = grouped.pivot(index=['pep_start', 'pep_end'], columns='hx_time', values='UptakeFraction').reset_index()

# # drop 
# grouped.head()


# %%

# # print entire dataframe
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# print(grouped)

# %%
# # conver to HDXer format ie start, end, exposure_1, exposure_2 

# # change Start to ResStr and End to ResEnd
# hdx = grouped.rename(columns={'pep_start': 'ResStr', 'pep_end': 'ResEnd'})

# # drop the exposure column
# hdx.columns.name = None

# print(hdx)


# %%

# hdx = hdx.round(5)
# hdx.to_csv(os.path.join("raw_data", "MBP", 'MBP_wt1.dat'), sep=' ', index=False)


# %%
# segs = hdx[['ResStr', 'ResEnd']].drop_duplicates().sort_values(by=['ResStr', 'ResEnd']).reset_index(drop=True)


# %%

# # convert to list of tuples
# segs = [tuple(x) for x in segs.values]

# print(segs)


# %%


# # write list as new lines with space delimiter
# with open(os.path.join("raw_data", "MBP", 'MBP_wt1_segs.txt'), 'w') as f:
#     for item in segs:
#         f.write("%s\n" % ' '.join(map(str, item)))

# %%

# BPTI_dir = "/Users/alexi/Library/CloudStorage/OneDrive-Nexus365/Rotation_Projects/Rotation_3/Project/ValDX/raw_data/MBP/MaltoseBindingProtein"
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
    # BPTI_dir = "/Users/alexi/Library/CloudStorage/OneDrive-Nexus365/Rotation_Projects/Rotation_3/Project/ValDX/raw_data/MBP/MaltoseBindingProtein"
    BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/MBP/MaltoseBindingProtein"
    # BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI"

    sim_name = 'MBPwt_AF'
    os.listdir(BPTI_dir)

    segs_name = "MBP_wt1_segs.txt"
    segs_path = os.path.join(BPTI_dir, segs_name)

    hdx_name = "MBP_wt1.dat"
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
        fixer = PDBFixer(os.path.join(sim_dir, pdb))
        fixer.addMissingHydrogens(7.0)
        H_pdb_name = pdb.replace('.pdb', '_H.pdb')
        PDBFile.writeFile(fixer.topology, fixer.positions, open(os.path.join(H_sim_dir, H_pdb_name), 'w'), keepIds=True)

    pdb_list = [f for f in os.listdir(H_sim_dir) if f.endswith('.pdb')]


    top_path = "/home/alexi/Documents/ValDX/raw_data/MBP/MBP_wt_protonated.pdb"

    # pdb_paths = [os.path.join(H_sim_dir, i) for i in pdb_list]

    # print(top_path)
    # print(pdb_paths)


    # small_traj_name = top_path.replace(".pdb","_small.xtc")
    # small_traj_path = os.path.join(sim_dir, small_traj_name)

    # u = mda.Universe(top_path, pdb_paths)


        
    # with XTCWriter(small_traj_path, n_atoms=u.atoms.n_atoms) as W:
    #     for ts in u.trajectory:
    #             W.write(u.atoms)

    # traj_paths = [os.path.join(sim_dir, i) for i in os.listdir(sim_dir) if i.endswith(".pdb")]
    
    traj_paths = ["/home/alexi/Documents/ValDX/raw_data/MBP/MBP_wt_protonated.xtc"]

    print(traj_paths)
    return hdx_path, segs_path, rates_path, top_path, traj_paths, sim_name, expt_name, test_name


# %%
hdx_path, segs_path, rates_path, top_path, traj_paths, sim_name, expt_name, test_name = pre_process_main()

# %%
ic.disable()
settings.cluster_frac1 = 0.5
combined_analysis_dump, names, save_paths = VDX.run_refine_ensemble(system=test_name+"_mode",
                                                                        times=[30, 240, 1800, 14400],
                                                                        expt_name=expt_name,
                                                                        n_reps=2,
                                                                        split_mode='R3',
                                                                        # RW=True,
                                                                        hdx_path=hdx_path,
                                                                        segs_path=segs_path,
                                                                        traj_paths=traj_paths,
                                                                        top_path=top_path,
                                                                        modal_cluster=True)


# %%
combined_analysis_dump, names, save_paths = VDX.run_refine_ensemble(system=test_name+"_mean",
                                                                        times=[30, 240, 1800, 14400],
                                                                        expt_name=expt_name,
                                                                        n_reps=2,
                                                                        split_mode='R3',
                                                                        # RW=True,
                                                                        hdx_path=hdx_path,
                                                                        segs_path=segs_path,
                                                                        traj_paths=traj_paths,
                                                                        top_path=top_path,
                                                                        modal_cluster=False)
