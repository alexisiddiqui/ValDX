# %%
### ValDXer testing
import os
os.environ["HDXER_PATH"] = "/home/alexi/Documents/HDXer"
from ValDX.ValidationDX import ValDXer
from ValDX.VDX_Settings import Settings
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter
from icecream import ic

settings = Settings(name='BPTI_shaw')
settings.replicates = 3
settings.gamma_range = (1,8)
settings.train_frac = 0.5
settings.RW_exponent = [0]
settings.split_mode = 'R3'
settings.stride = 1000
# settings.HDXer_stride = 10000

settings.RW_do_reweighting = False
settings.RW_do_params = True
import pickle

VDX = ValDXer(settings)


# %%
settings.save_figs

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
    test_name = "BPTI_shaw1000"

    BPTI_dir = "/Users/alexi/Library/CloudStorage/OneDrive-Nexus365/Rotation_Projects/Rotation_3/Project/ValDX/raw_data/HDXer_tutorial/BPTI"
    BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI"
    # BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI"
    expt_dir = os.path.join(BPTI_dir, "BPTI_expt_data")

    os.listdir(expt_dir)

    segs_name = "BPTI_residue_segs.txt"
    segs_path = os.path.join(expt_dir, segs_name)

    hdx_name = "BPTI_expt_dfracs.dat"
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


    small_traj_name = traj_name.replace(".xtc","_small.xtc")
    small_traj_path = os.path.join(sim_dir, small_traj_name)

    u = mda.Universe(top_path, traj_paths)

    
        
    with XTCWriter(small_traj_path, n_atoms=u.atoms.n_atoms) as W:
        for ts in u.trajectory[:5]:
                W.write(u.atoms)

    top_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/AF_sample_quick/P00974_protonated.pdb"
    traj_paths = ["/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/AF_sample_quick/P00974_protonated.xtc"]


    top_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/SHAW/bpti.pdb"
    traj_paths =["/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/SHAW/reduced_BPTI_SHAW_stride_1000.xtc"]

    return hdx_path, segs_path, rates_path, top_path, traj_paths, sim_name, expt_name, test_name


# %%
hdx_path, segs_path, rates_path, top_path, traj_paths, sim_name, expt_name, test_name = pre_process_main_BPTI()

# %%


# %%


# %%


# %%



# %%
# combined_analysis_dump, names, save_paths = VDX.run_benchmark_ensemble(system=test_name,
#                                                                         times=[0.167, 1, 10, 120],
#                                                                         expt_name=expt_name,
#                                                                         n_reps=1,
#                                                                         # split_modes=['r'],
#                                                                         # RW=True,
#                                                                         optimise=False,
#                                                                         hdx_path=hdx_path,
#                                                                         segs_path=segs_path,
#                                                                         traj_paths=traj_paths,
#                                                                         top_path=top_path)

                                                                        

# %%
# ic.disable()
# VDX.settings.cluster_frac1 = 0.1
# VDX.settings.cluster_size2 = 10
# # break
# combined_analysis_dump, names, save_paths = VDX.run_refine_ensemble(system=test_name,
#                                                                         times=[0.167, 1, 10, 120],
#                                                                         expt_name=expt_name,
#                                                                         n_reps=2,
#                                                                         split_mode='R3',
#                                                                         # RW=True,
#                                                                         hdx_path=hdx_path,
#                                                                         segs_path=segs_path,
#                                                                         traj_paths=traj_paths,
#                                                                         top_path=top_path)

# %%
ic.disable()
VDX.settings.cluster_frac1 = 0.5
VDX.settings.cluster_size2 = 50
combined_analysis_dump, names, save_paths = VDX.run_refine_ensemble(system=test_name+"_modal_cluster",
                                                                        times=[0.167, 1, 10, 120],
                                                                        expt_name=expt_name,
                                                                        n_reps=2,
                                                                        split_mode='R3',
                                                                        # RW=True,
                                                                        
                                                                        hdx_path=hdx_path,
                                                                        segs_path=segs_path,
                                                                        traj_paths=traj_paths,
                                                                        top_path=top_path,
                                                                        modal_cluster=True)



combined_analysis_dump, names, save_paths = VDX.run_refine_ensemble(system=test_name+"_mean_cluster",
                                                                        times=[0.167, 1, 10, 120],
                                                                        expt_name=expt_name,
                                                                        n_reps=2,
                                                                        split_mode='R3',
                                                                        # RW=True,
                                                                        
                                                                        hdx_path=hdx_path,
                                                                        segs_path=segs_path,
                                                                        traj_paths=traj_paths,
                                                                        top_path=top_path,
                                                                        modal_cluster=False)

