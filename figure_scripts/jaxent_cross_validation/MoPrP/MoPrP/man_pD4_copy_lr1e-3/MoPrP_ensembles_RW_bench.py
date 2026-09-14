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
def pre_process_main_MoPrP():
    # MoPrP data
    expt_name = "Experimental"
    test_name = "MoPrP_af_rank1"
    test_names = [
        # "MoPrP_TFES",
        # "MoPrP_af_dirty",
        "MoPrP_af_clean",
        # "MoPrP_shaw_400",
        # "MoPrP_MD_Bad",
        # "MoPrP_MD_Good",
        # "MoPrP_MD_Good+Bad",
    ]

    MoPrP_dir = "/home/alexi/Documents/Library/CloudStorage/OneDrive-Nexus365/Rotation_Projects/Rotation_3/Project/ValDX/raw_data/HDXer_tutorial/MoPrP"
    MoPrP_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/MoPrP"
    # MoPrP_dir = "/data/localhost/not-backed-up/hussain/ValDX/raw_data/HDXer_tutorial/MoPrP"
    # MoPrP_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/MoPrP"
    expt_dir = os.path.join(MoPrP_dir, "MoPrP_expt_data")

    # os.listdir(expt_dir)

    segs_name = "MoPrP_residue_segs_trimmed.txt"
    segs_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_cross_validation/MoPrP/MoPrP/output/MoPrP_segments.txt"

    hdx_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_cross_validation/MoPrP/MoPrP/output/MoPrP_dfrac.dat"
    print(hdx_path)

    rates_name = "MoPrP_Intrinsic_rates.dat"
    rates_path = os.path.join(expt_dir, rates_name)
    sim_name = "MoPrP_MD"

    sim_dir = os.path.join(MoPrP_dir, "MoPrP_simulations")

    md_reps = 1
    rep_dirs = ["Run_" + str(i + 1) for i in range(md_reps)]

    top_name = "/home/alexi/Documents/xFold_Sampling/af_sample/MoPrP_max_plddt_4334.pdb"

    top_path = os.path.join(sim_dir, rep_dirs[0], top_name)

    traj_name = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_cross_validation/MoPrP/MoPrP/all_clusters.xtc"

    traj_paths = [os.path.join(sim_dir, rep_dir, traj_name) for rep_dir in rep_dirs]

    print(top_path)
    print(traj_paths)

    topology_path = "/home/alexi/Documents/xFold_Sampling/af_sample/MoPrP_max_plddt_4334.pdb"
    # AF2 filtered/clean
    clean_traj_path = "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc"
    # AF2 MSAss/dirty/unfiltered
    dirty_traj_path = "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc"

    top_paths = [
        topology_path,
        # dirty_top_path,
        # clean_top_path,
        # shaw_top_path,
        # badMD_top_path,
        # goodMD_top_path,
        # goodMD_top_path,
    ]
    traj_paths = [
        # trajectory_path,
        clean_traj_path,
        # dirty_traj_path,
        # shaw_traj_paths[0],
        # badMD_traj_path[0],
        # goodMD_traj_path[0],
        # [goodMD_traj_path[0], badMD_traj_path[0]],
    ]

    return hdx_path, segs_path, rates_path, top_paths, traj_paths, sim_name, expt_name, test_names


# %%
hdx_path, segs_path, rates_path, top_paths, traj_paths, sim_name, expt_name, test_names = (
    pre_process_main_MoPrP()
)

# %%


# %%


# %%


# %%

times = [
    0.08,
    0.33,
    0.67,
    1.00,
    5.00,
    10.00,
    20.00,
    30.00,
    45.00,
    60.00,
    160.00,
    240.00,
    390.00,
    750.00,
    1440.00,
]
for idx, (test_name, top_path, traj_paths) in enumerate(
    zip(test_names[:1], top_paths[:1], traj_paths[:1])
):
    # if idx < 5:
    #     continue

    settings = Settings(name=test_name)
    # settings.replicates = 2
    settings.gamma_range = (1, 10)

    settings.train_frac = 0.5
    settings.RW_exponent = [-1, 0, 1]
    settings.split_mode = ["R3", "Sp"]
    VDX = ValDXer(settings)

    # run RW across all splits

    combined_analysis_dump, names, save_paths = VDX.run_benchmark_ensemble(
        system=test_name,
        times=times,
        split_modes=["R3", "Sp"],
        expt_name=expt_name,
        n_reps=3,
        RW=True,
        hdx_path=hdx_path,
        segs_path=segs_path,
        traj_paths=[traj_paths],
        top_path=top_path,
    )


end_time = time.time()

print(f"Time taken: {end_time - start_time} s")
# time in minutes
print(f"Time taken: {(end_time - start_time) / 60} min")
