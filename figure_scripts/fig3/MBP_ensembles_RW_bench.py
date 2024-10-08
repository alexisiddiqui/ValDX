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
# from pdbfixer import PDBFixer
# # from openmm.app import PDBFile
# settings = Settings(name='MBP')
# # settings.replicates = 1
# settings.gamma_range = (1,8)
# settings.train_frac = 0.5
# settings.RW_exponent = [0]
# settings.split_mode = 'R3'
# # settings.stride = 1000
# # settings.HDXer_stride = 10000

# settings.RW_do_reweighting = False
# settings.RW_do_params = True
import pickle

# VDX = ValDXer(settings)
# expt_name = 'Experimental'
# test_name = "MBP_af_dirty"

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


def MD_traj_to_interval_paths(top_path,
                              traj_path,
                              n_intervals=10,
                              n_reps=None,
                              out_path=None):
    

    top_name = os.path.basename(top_path).replace(".pdb", "")        
    
    if n_reps is None:
        # extract the number of replicates from the top_name
        split = top_name.split("_")
        # find split that starts with "r" 
        for s in split:
            if s.startswith("r"):
                if s[1:].isdigit():
                    n_reps = int(s[1:])
                    break
        
    
    if out_path is None:
        out_path = os.path.join(os.path.dirname(traj_path), 'time_intervals')
        
    os.makedirs(out_path, exist_ok=True)

    # load the trajectory

    u = mda.Universe(top_path, traj_path)

    length = len(u.trajectory)

    # check that the number of frames is divisible by n_reps

    assert length % n_reps == 0, f"Number of frames {length} is not divisible by n_reps {n_reps}"
    
    traj_length = length / n_reps

    # round down interval length to the nearest integer
    interval_length = int(traj_length / n_intervals)


    # universe represents a concatenated trajectory for each replicate
    # when slicing into intervals the frames must be selected from each replicate
    # in a way that the intervals are continuous in time
    # create a new trajectory for each interval

    interval_indexes = {i: [] for i in range(n_intervals)}

    for i in range(n_intervals):
        
        start = i * interval_length
        end = ((i+1) * interval_length)

        for j in range(n_reps):
            interval_indexes[i] += list(range(int(j*traj_length + start), int(j*traj_length + end)))

    # create a new trajectory for each interval
    print(interval_indexes[0])

    interval_names = [f"{top_name}_nI{n_intervals}_interval{i}_len{n_intervals*interval_length}"+".xtc" for i in range(n_intervals)]


    new_traj_paths = []

    for i, indexes in interval_indexes.items():
        new_traj_path = os.path.join(out_path, interval_names[i])
        new_traj_paths.append([new_traj_path])


    top_paths = [top_path]*n_intervals

    return new_traj_paths, top_paths




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
    test_names = ["MBP_af_dirty", "MBP_af_clean", "MBP_MD_Bad", "MBP_MD_Good", "MBP_MD_Good+Bad"]
    expt_name = 'Experimental'

    sim_name = 'MBPwt_AF'
    os.listdir(BPTI_dir)

    segs_name = "MBP_wt1_segs.txt"
    segs_path = os.path.join(BPTI_dir, segs_name)

    hdx_name = "MBP_wt1_clean.dat"
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


    dirty_top_path = "/home/alexi/Documents/ValDX/raw_data/MBP/MBP_wt_1_af_sample_127_10000_protonated.pdb"

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
    
    dirty_traj_paths = ["/home/alexi/Documents/ValDX/raw_data/MBP/MBP_wt_1_af_sample_127_10000_protonated.xtc"]

    clean_top_path = dirty_top_path
    clean_traj_paths = [dirty_traj_paths[0].replace(".xtc", "_all_filtered.xtc")]

    badMD_top_path = "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/MBP/BadMD_MBP_r5_15010_concatenated.pdb"
    badMD_traj_path = [badMD_top_path.replace(".pdb", ".xtc")]

    goodMD_top_path = "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/MBP/GoodMD_MBP_r10_10010_concatenated.pdb"
    goodMD_traj_path = [goodMD_top_path.replace(".pdb", ".xtc")]

    top_paths = [dirty_top_path, clean_top_path, badMD_top_path, goodMD_top_path, goodMD_top_path]
    traj_paths = [dirty_traj_paths[0], clean_traj_paths[0], badMD_traj_path[0], goodMD_traj_path[0], [goodMD_traj_path[0], badMD_traj_path[0]]]

    min_interval_size=500
    confidence_intervals = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), ("top", min_interval_size), ("bottom", min_interval_size)]
    str_confidence_intervals = [f"{i}_{j}" for (i,j) in confidence_intervals]
    conf_interval_names = [f"MBP_af_conf{i}" for i in str_confidence_intervals]
    conf_interval_traj_names = [dirty_traj_path.replace(".xtc", f"_{name}.xtc") for name in str_confidence_intervals for dirty_traj_path in dirty_traj_paths]
    conf_dir = "af_confidence_intervals"
    conf_interval_paths = [os.path.join(os.path.dirname(dirty_top_path), conf_dir, os.path.basename(name)) for name in conf_interval_traj_names]

    BadMD_interval_traj_paths, BadMD_interval_top_paths = MD_traj_to_interval_paths(badMD_top_path, badMD_traj_path[0])
    BadMD_interval_test_names = [f"MBP_MD_Bad-Int{i}" for i in range(10)]
    GoodMD_interval_traj_paths, GoodMD_interval_top_paths = MD_traj_to_interval_paths(goodMD_top_path, goodMD_traj_path[0])
    GoodMD_interval_test_names = [f"MBP_MD_Good-Int{i}" for i in range(10)]


    # test_names = test_names + conf_interval_names
    # top_paths = top_paths + [dirty_top_path]*len(conf_interval_names)
    # traj_paths = traj_paths + conf_interval_paths

    # top_paths = BadMD_interval_top_paths + GoodMD_interval_top_paths 
    # traj_paths = BadMD_interval_traj_paths + GoodMD_interval_traj_paths
    # test_names = BadMD_interval_test_names + GoodMD_interval_test_names


    return hdx_path, segs_path, rates_path, top_paths, traj_paths, sim_name, expt_name, test_names


# %%
hdx_path, segs_path, rates_path, top_paths, traj_paths, sim_name, expt_name, test_names = pre_process_main()


           

times = [0.5,	4.0,	30.0]


for idx ,(test_name, top_path, traj_paths) in enumerate(zip(test_names, top_paths, traj_paths)):

    # if idx > 4:
    #     continue
    settings = Settings(name=test_name)
    # settings.replicates = 2
    settings.gamma_range = (1,8)
    settings.train_frac = 0.5
    settings.RW_exponent = [0]
    # settings.split_mode = 'R3'

    VDX = ValDXer(settings)


    # run RW across all splits

    combined_analysis_dump, names, save_paths = VDX.run_cluster_benchmark_ensemble(system=test_name,
                                                                            times=times,
                                                                            expt_name=expt_name,
                                                                            n_reps=3,

                                                                            RW=True,
                                                                            hdx_path=hdx_path,
                                                                            segs_path=segs_path,
                                                                            traj_paths=[traj_paths],
                                                                            top_path=top_path)

                                                                            

