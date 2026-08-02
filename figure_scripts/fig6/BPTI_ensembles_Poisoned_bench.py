# %%
### ValDXer testing
import os
# os.environ["HDXER_PATH"] = "/homes/hussain/HDXer"
os.environ["HDXER_PATH"] = "/home/alexi/Documents/HDXer"

import sys
sys.path.append("/home/alexi/Documents/ValDX/")


from ValDX.ValidationDX import ValDXer
from ValDX.VDX_Settings import Settings
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter
from icecream import ic

# settings.stride = 1000
# # settings.HDXer_stride = 10000

# settings.RW_do_reweighting = False
# settings.RW_do_params = True
import pickle



# %%

# %%
# import subprocess
# from ValDX.helpful_funcs import conda_to_env_dict

# # Assuming settings.HDXer_env contains the name of your Conda environment
# env_path = conda_to_env_dict(settings.HDXer_env)

# command = "echo $HDXER_PATH"
# print("command:", command)

# # Run the command in the subprocess
# output = subprocess.run(command, shell=True, env=env_path, capture_output=True, text=True)

# # Capture and print the standard output (stdout)
# hdxer_path = output.stdout.strip()  # .strip() removes any trailing newline
# print("HDXER_PATH:", hdxer_path)


# def MD_traj_to_interval_paths(top_path,
#                               traj_path,
#                               n_intervals=10,
#                               n_reps=None,
#                               out_path=None):
    

#     top_name = os.path.basename(top_path).replace(".pdb", "")        
    
#     if n_reps is None:
#         # extract the number of replicates from the top_name
#         split = top_name.split("_")
#         # find split that starts with "r" 
#         for s in split:
#             if s.startswith("r"):
#                 if s[1:].isdigit():
#                     n_reps = int(s[1:])
#                     break
        
    
#     if out_path is None:
#         out_path = os.path.join(os.path.dirname(traj_path), 'time_intervals')
        
#     os.makedirs(out_path, exist_ok=True)

#     # load the trajectory

#     u = mda.Universe(top_path, traj_path)

#     length = len(u.trajectory)

#     # check that the number of frames is divisible by n_reps

#     assert length % n_reps == 0, f"Number of frames {length} is not divisible by n_reps {n_reps}"
    
#     traj_length = length / n_reps

#     # round down interval length to the nearest integer
#     interval_length = int(traj_length / n_intervals)


#     # universe represents a concatenated trajectory for each replicate
#     # when slicing into intervals the frames must be selected from each replicate
#     # in a way that the intervals are continuous in time
#     # create a new trajectory for each interval

#     interval_indexes = {i: [] for i in range(n_intervals)}

#     for i in range(n_intervals):
        
#         start = i * interval_length
#         end = ((i+1) * interval_length)

#         for j in range(n_reps):
#             interval_indexes[i] += list(range(int(j*traj_length + start), int(j*traj_length + end)))

#     # create a new trajectory for each interval
#     print(interval_indexes[0])

#     interval_names = [f"{top_name}_nI{n_intervals}_interval{i}_len{n_intervals*interval_length}"+".xtc" for i in range(n_intervals)]


#     new_traj_paths = []

#     for i, indexes in interval_indexes.items():
#         new_traj_path = os.path.join(out_path, interval_names[i])
#         new_traj_paths.append([new_traj_path])


#     top_paths = [top_path]*n_intervals

#     return new_traj_paths, top_paths
def pre_process_main_BPTI():
    # BPTI experimental data paths
    BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI"
    expt_dir = os.path.join(BPTI_dir, "BPTI_expt_data")
    
    # Experimental data files
    segs_name = "BPTI_residue_segs_trimmed.txt"
    segs_path = os.path.join(expt_dir, segs_name)
    
    hdx_name = "BPTI_expt_dfracs_clean_trimmed.dat"
    hdx_path = os.path.join(expt_dir, hdx_name)
    
    rates_name = "BPTI_Intrinsic_rates.dat"
    rates_path = os.path.join(expt_dir, rates_name)
    
    # Base paths
    base_path = "/home/alexi/Documents/interpretable-hdxer/notebooks/Figure_5_Poisoned_Ensemble/combined_ensembles/BPTI"
    regular_MD_base = "/home/alexi/Documents/ValDX/raw_data/full_length_regular_MD"

    # Define ensembles and their properties
    ensembles = {
        # "AF2-MSAss": {
        #     "top": "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/P00974_60_1_af_sample_127_10000_protonated.pdb",
        # },
        # "AF2-Cleaned": {
        #     "top": "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/P00974_60_1_af_sample_127_10000_protonated.pdb"
        # },
        # "T-FES": {
        #     "top": "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/T-FES/BPTI/BPTI_overall_combined_stripped.pdb",
        # },
        "1Start-MD": {
            "top": f"{regular_MD_base}/BPTI_test_concatenated_stripped.pdb",
        }
    }
    
    # Poisoning methods and levels
    poison_methods = ["add-noise", "mix-coordinates", "shuffle-backbone-protons"]
    short_poison_methods = {"add-noise": "noise", "mix-coordinates": "mix", "shuffle-backbone-protons": "shuffle"}
    poison_levels = [0, 1, 10, 20, 50, 100, 500]
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
                test_name = f"BPTI_{ensemble_name}_{short_method}_p{level}"
                test_names.append(test_name)
                
                # Add topology path
                top_paths.append(ensemble_data["top"])
                
                # Create trajectory path - Note: no longer wrapping in list
                traj_name = f"BPTI_{method}_poison{level}.xtc"
                traj_path = os.path.join(base_path, ensemble_name, traj_name)
                traj_paths.append(traj_path)  # Store as string instead of list
    
    sim_name = 'BPTI_Poisoned'
    expt_name = 'Experimental'
    
    return hdx_path, segs_path, rates_path, top_paths, traj_paths, sim_name, expt_name, test_names


# %%
hdx_path, segs_path, rates_path, top_paths, traj_paths, sim_name, expt_name, test_names = pre_process_main_BPTI()

# %%


# %%


# %%


# %%
times=[0.167, 1, 10]

for idx ,(test_name, top_path, traj_paths) in enumerate(zip(test_names, top_paths, traj_paths)):
    # if idx < 5:
    #     continue

    settings = Settings(name=test_name)
    # settings.replicates = 2
    settings.gamma_range = (1,8)
    settings.train_frac = 0.5
    settings.RW_exponent = [0]
    # settings.split_mode = 'R3'

    VDX = ValDXer(settings)


    # run RW across all splits

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

                                                                            

