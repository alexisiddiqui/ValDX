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


# %% [markdown]
# 

# %%
def pre_process_main_BPTI():
    # BPTI data
    expt_name = 'Experimental'
    test_name = "BPTI_af_rank1"
    test_names = ["BPTI_af_dirty", "BPTI_af_clean", "BPTI_shaw_400"]

    BPTI_dir = "/Users/alexi/Library/CloudStorage/OneDrive-Nexus365/Rotation_Projects/Rotation_3/Project/ValDX/raw_data/HDXer_tutorial/BPTI"
    BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI"
    # BPTI_dir = "/data/localhost/not-backed-up/hussain/ValDX/raw_data/HDXer_tutorial/BPTI"
    # BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI"
    expt_dir = os.path.join(BPTI_dir, "BPTI_expt_data")

    os.listdir(expt_dir)

    segs_name = "BPTI_residue_segs_trimmed.txt"
    segs_path = os.path.join(expt_dir, segs_name)

    hdx_name = "BPTI_expt_dfracs_clean_trimmed.dat"
    hdx_path = os.path.join(expt_dir, hdx_name)
    print(hdx_path)

    rates_name = "BPTI_Intrinsic_rates.dat"
    rates_path = os.path.join(expt_dir, rates_name)
    sim_name = 'BPTI_MD'

    sim_dir = os.path.join(BPTI_dir, "BPTI_simulations")

    os.listdir(sim_dir)

    md_reps = 1
    rep_dirs = ["Run_"+str(i+1) for i in range(md_reps)]

    top_name = "bpti_5pti_eq6_protonly.gro"

    top_path = os.path.join(sim_dir, rep_dirs[0], top_name)

    traj_name = "bpti_5pti_reimg_protonly.xtc"

    traj_paths = [os.path.join(sim_dir, rep_dir, traj_name) for rep_dir in rep_dirs]

    print(top_path)
    print(traj_paths)


    dirty_top_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10000_protonated.pdb"
    # top_path =  "/data/localhost/not-backed-up/hussain/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10001_protonated.pdb"
    dirty_traj_paths = ["/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10000_protonated.xtc"]
    # traj_paths = ["/data/localhost/not-backed-up/hussain/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10001_protonated.xtc"]


    clean_top_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10001_protonated.pdb"
    clean_traj_paths = ["/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10000_protonated_all_filtered.xtc"]


    shaw_top_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/SHAW/bpti.pdb"
    shaw_traj_paths =["/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/SHAW/reduced_BPTI_SHAW_stride_400.xtc"]


    top_paths = [dirty_top_path, clean_top_path, shaw_top_path]
    traj_paths = [dirty_traj_paths[0], clean_traj_paths[0], shaw_traj_paths[0]]

    min_interval_size=500
    confidence_intervals = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), ("top", min_interval_size), ("bottom", min_interval_size)]
    str_confidence_intervals = [f"{i}_{j}" for (i,j) in confidence_intervals]
    conf_interval_names = [f"BPTI_af_conf{i}" for i in str_confidence_intervals]
    conf_interval_traj_names = [dirty_traj_path.replace(".xtc", f"_{name}.xtc") for name in str_confidence_intervals for dirty_traj_path in dirty_traj_paths]
    conf_dir = "af_confidence_intervals"
    conf_interval_paths = [os.path.join(os.path.dirname(dirty_top_path), conf_dir, os.path.basename(name)) for name in conf_interval_traj_names]

    test_names = test_names + conf_interval_names
    top_paths = top_paths + [dirty_top_path]*len(conf_interval_names)
    traj_paths = traj_paths + conf_interval_paths

    return hdx_path, segs_path, rates_path, top_paths, traj_paths, sim_name, expt_name, test_names





# %%
hdx_path, segs_path, rates_path, top_paths, traj_paths, sim_name, expt_name, test_names = pre_process_main_BPTI()

# %%


# %%


# %%


# %%
times=[0.167, 1, 10]

for (test_name, top_path, traj_paths) in zip(test_names, top_paths, traj_paths):

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

                                                                            

