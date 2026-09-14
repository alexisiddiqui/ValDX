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
# from openmm.app import PDBFile

# settings = Settings(name='HOIP')
# # settings.replicates = 1
# settings.gamma_range = (1,8)
# settings.train_frac = 0.5
# settings.RW_exponent = [0]
# settings.split_mode = 'R3'
# settings.stride = 1000
# settings.HDXer_stride = 10000

# settings.RW_do_reweighting = True
# settings.RW_do_params = False
import pickle

# VDX = ValDXer(settings)
# expt_name = 'Experimental'
# test_name = "HOIP_af_dirty"
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



# %%
def pre_process_main():
    # HOIP experimental data paths
    HOIP_dir = "/home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo"
    expt_dir = HOIP_dir
    
    # Experimental data files
    segs_name = "HOIP_APO_segs_trimmed.txt"
    segs_path = os.path.join(expt_dir, segs_name)
    
    hdx_name = "HOIP_apo_clean_trimmed.dat"
    hdx_path = os.path.join(expt_dir, hdx_name)
    
    rates_name = "out__train_MD_Simulated_1Intrinsic_rates.dat"
    rates_path = os.path.join(expt_dir, rates_name)
    
    # Base paths
    base_path = "/home/alexi/Documents/interpretable-hdxer/notebooks/Figure_5_Poisoned_Ensemble/combined_ensembles/HOIP"
    
    regular_MD_base = "/home/alexi/Documents/ValDX/raw_data/full_length_regular_MD"

    # Define ensembles and their properties
    ensembles = {
        "AF2-MSAss": {
            "top": "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/HOIP_apo697_1_af_sample_127_10000_protonated.pdb",
        },
        # "AF2-Cleaned": {
        #     "top": "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/HOIP_apo697_1_af_sample_127_10000_protonated.pdb"
        # },
        # "T-FES": {
        #     "top": "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/T-FES/HOIP/HOIP_overall_combined_stripped.pdb",
        # },
        # "1Start-MD": {
        #     "top":f"{regular_MD_base}/HOIP_test_concatenated_stripped.pdb"
        # }
    }  
    
    
    # Poisoning methods and levels
    poison_methods = ["add-noise", "mix-coordinates", "shuffle-backbone-protons"]

    short_poison_methods = {"add-noise": "noise", "mix-coordinates": "mix", "shuffle-backbone-protons": "shuffle"}
    poison_levels = [20, 50, 100 ]
    poison_levels = [0, 1, 10, 20, 50, 100, 500]

    # Initialize lists to store all combinations
    test_names = []
    top_paths = []
    traj_paths = []
    
    # Generate all combinations of ensembles, methods, and levels
    for ensemble_name, ensemble_data in ensembles.items():
        for method in poison_methods:
            for level in poison_levels:
                # Create test name
                short_method = short_poison_methods[method]
                test_name = f"HOIP_{ensemble_name}_{short_method}_p{level}"
                test_names.append(test_name)
                
                # Add topology path
                top_paths.append(ensemble_data["top"])
                
                # Create trajectory path
                traj_name = f"HOIP_{method}_poison{level}.xtc"
                traj_path = os.path.join(base_path, ensemble_name, traj_name)
                traj_paths.append(traj_path)  # Store as string instead of list
    
    sim_name = 'HOIP_Poisoned'
    expt_name = 'Experimental'
    
    return hdx_path, segs_path, rates_path, top_paths, traj_paths, sim_name, expt_name, test_names

# %%
hdx_path, segs_path, rates_path, top_paths, traj_paths, sim_name, expt_name, test_names = pre_process_main()
times = [0.5, 5.0]

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

    try:
        combined_analysis_dump, names, save_paths = VDX.run_benchmark_ensemble(system=test_name,
                                                                                times=times,
                                                                                expt_name=expt_name,
                                                                                n_reps=3,
                                                                                split_modes=['R3', "Sp"],
                                                                                RW=True,
                                                                                hdx_path=hdx_path,
                                                                                segs_path=segs_path,
                                                                                traj_paths=[traj_paths],
                                                                                top_path=top_path)
    except Exception as e:
        print(e)
        continue

                                                                            

