# %%
### ValDXer testing
import os
import time

start_time = time.time()

# os.environ["HDXER_PATH"] = "/homes/hussain/HDXer"
os.environ["HDXER_PATH"] = "/home/alexi/Documents/HDXer"

import sys

sys.path.append("/home/alexi/Documents/ValDX/")


import MDAnalysis as mda

from ValDX.ValidationDX import ValDXer
from ValDX.VDX_Settings import Settings

# settings.stride = 1000
# # settings.HDXer_stride = 10000

# settings.RW_do_reweighting = False
# settings.RW_do_params = True

# change the path to this script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


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


def MD_traj_to_interval_paths(top_path, traj_path, n_intervals=10, n_reps=None, out_path=None):
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
        out_path = os.path.join(os.path.dirname(traj_path), "time_intervals")

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
        end = (i + 1) * interval_length

        for j in range(n_reps):
            interval_indexes[i] += list(
                range(int(j * traj_length + start), int(j * traj_length + end))
            )

    # create a new trajectory for each interval
    print(interval_indexes[0])

    interval_names = [
        f"{top_name}_nI{n_intervals}_interval{i}_len{n_intervals * interval_length}" + ".xtc"
        for i in range(n_intervals)
    ]

    new_traj_paths = []

    for i, indexes in interval_indexes.items():
        new_traj_path = os.path.join(out_path, interval_names[i])
        new_traj_paths.append([new_traj_path])

    top_paths = [top_path] * n_intervals

    return new_traj_paths, top_paths


# %%
def pre_process_main_BPTI():
    # BPTI data
    expt_name = "Experimental"
    test_name = "TeaA_af_rank1"
    # test_names = [
    #     "BPTI_TFES",
    #     "BPTI_af_dirty",
    #     "BPTI_af_clean",
    #     "BPTI_shaw_400",
    #     "BPTI_MD_Bad",
    #     "BPTI_MD_Good",
    #     "BPTI_MD_Good+Bad",
    # ]

    test_names = ["TeaA_auto_VAL"]
    test_names = ["TeaA_ISO_bi", "TeaA_ISO_tri"]

    BPTI_dir = "/home/alexi/Documents/Library/CloudStorage/OneDrive-Nexus365/Rotation_Projects/Rotation_3/Project/ValDX/raw_data/HDXer_tutorial/BPTI"
    BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI"
    # BPTI_dir = "/data/localhost/not-backed-up/hussain/ValDX/raw_data/HDXer_tutorial/BPTI"
    # BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI"
    expt_dir = os.path.join(BPTI_dir, "BPTI_expt_data")

    os.listdir(expt_dir)

    segs_name = "BPTI_residue_segs_trimmed.txt"
    segs_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/TeaA/output/mixed_60-40_artificial_expt_resfracs_TeaA_segs.txt"

    hdx_name = "BPTI_expt_dfracs_clean_trimmed.dat"
    hdx_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/TeaA/output/mixed_60-40_artificial_expt_resfracs_TeaA_dfrac.dat"
    print(hdx_path)

    rates_name = "BPTI_Intrinsic_rates.dat"
    rates_path = os.path.join(expt_dir, rates_name)
    sim_name = "BPTI_MD"

    sim_dir = os.path.join(BPTI_dir, "BPTI_simulations")

    os.listdir(sim_dir)

    md_reps = 1
    rep_dirs = ["Run_" + str(i + 1) for i in range(md_reps)]

    top_name = "bpti_5pti_eq6_protonly.gro"

    top_path = os.path.join(sim_dir, rep_dirs[0], top_name)

    traj_name = "bpti_5pti_reimg_protonly.xtc"

    traj_paths = [os.path.join(sim_dir, rep_dir, traj_name) for rep_dir in rep_dirs]

    print(top_path)
    print(traj_paths)

    dirty_top_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10000_protonated.pdb"
    # top_path =  "/data/localhost/not-backed-up/hussain/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10001_protonated.pdb"
    dirty_traj_paths = [
        "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10000_protonated.xtc"
    ]
    # traj_paths = ["/data/localhost/not-backed-up/hussain/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10001_protonated.xtc"]

    clean_top_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10001_protonated.pdb"
    clean_traj_paths = [
        "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10000_protonated_all_filtered.xtc"
    ]

    shaw_top_path = (
        "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/SHAW/bpti.pdb"
    )
    shaw_traj_paths = [
        "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/SHAW/reduced_BPTI_SHAW_stride_400.xtc"
    ]

    badMD_top_path = (
        "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BPTI/BadMD_BPTI_r5_15010_concatenated.pdb"
    )
    badMD_traj_path = [badMD_top_path.replace(".pdb", ".xtc")]

    goodMD_top_path = "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BPTI/GoodMD_BPTI_r10_10010_concatenated.pdb"
    goodMD_traj_path = [goodMD_top_path.replace(".pdb", ".xtc")]

    topology_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_ref_open_state.pdb"
    bi_trajectory_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_filtered.xtc"
    tri_trajectory_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_initial_sliced.xtc"

    top_paths = [
        topology_path,
        topology_path,
        # dirty_top_path,
        # clean_top_path,
        # shaw_top_path,
        # badMD_top_path,
        # goodMD_top_path,
        # goodMD_top_path,
    ]
    traj_paths = [
        bi_trajectory_path,
        tri_trajectory_path,
        # dirty_traj_paths[0],
        # clean_traj_paths[0],
        # shaw_traj_paths[0],
        # badMD_traj_path[0],
        # goodMD_traj_path[0],
        # [goodMD_traj_path[0], badMD_traj_path[0]],
    ]

    min_interval_size = 500
    confidence_intervals = [
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 1.0),
        ("top", min_interval_size),
        ("bottom", min_interval_size),
    ]
    str_confidence_intervals = [f"{i}_{j}" for (i, j) in confidence_intervals]
    conf_interval_names = [f"BPTI_af_conf{i}" for i in str_confidence_intervals]
    conf_interval_traj_names = [
        dirty_traj_path.replace(".xtc", f"_{name}.xtc")
        for name in str_confidence_intervals
        for dirty_traj_path in dirty_traj_paths
    ]
    conf_dir = "af_confidence_intervals"
    conf_interval_paths = [
        os.path.join(os.path.dirname(dirty_top_path), conf_dir, os.path.basename(name))
        for name in conf_interval_traj_names
    ]

    BadMD_interval_traj_paths, BadMD_interval_top_paths = MD_traj_to_interval_paths(
        badMD_top_path, badMD_traj_path[0]
    )
    BadMD_interval_test_names = [f"BPTI_MD_Bad-Int{i}" for i in range(10)]
    GoodMD_interval_traj_paths, GoodMD_interval_top_paths = MD_traj_to_interval_paths(
        goodMD_top_path, goodMD_traj_path[0]
    )
    GoodMD_interval_test_names = [f"BPTI_MD_Good-Int{i}" for i in range(10)]

    # test_names = test_names + conf_interval_names
    # top_paths = top_paths + [dirty_top_path]*len(conf_interval_names)
    # traj_paths = traj_paths + conf_interval_paths

    # top_paths = BadMD_interval_top_paths + GoodMD_interval_top_paths
    # traj_paths = BadMD_interval_traj_paths + GoodMD_interval_traj_paths
    # test_names = BadMD_interval_test_names + GoodMD_interval_test_names

    return hdx_path, segs_path, rates_path, top_paths, traj_paths, sim_name, expt_name, test_names


# %%
hdx_path, segs_path, _, top_paths, traj_paths, sim_name, expt_name, test_names = (
    pre_process_main_BPTI()
)

# %# %%


# %%


# %%
times = [0.167, 1, 10, 60, 120]

for idx, (test_name, top_path, traj_paths) in enumerate(zip(test_names, top_paths, traj_paths)):
    # if idx < 5:
    #     continue

    settings = Settings(name=test_name)
    # settings.replicates = 2
    settings.gamma_range = (1, 10)
    settings.train_frac = 0.5
    settings.RW_exponent = [-1, 0, 1]
    settings.split_mode = "r"

    VDX = ValDXer(settings)

    # run RW across all splits

    combined_analysis_dump, names, save_paths = VDX.run_benchmark_ensemble(
        system=test_name,
        times=times,
        expt_name=expt_name,
        n_reps=3,
        RW=True,
        BV=True,
        hdx_path=hdx_path,
        split_modes=["R3","Sp"],
        segs_path=segs_path,
        traj_paths=[traj_paths],
        top_path=top_path,
    )


end_time = time.time()

print(f"Time taken: {end_time - start_time} s")
# time in minutes
print(f"Time taken: {(end_time - start_time) / 60} min")
