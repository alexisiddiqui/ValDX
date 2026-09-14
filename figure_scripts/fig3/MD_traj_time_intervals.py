
import os 
import MDAnalysis as mda
import numpy as np
import json


def MD_traj_to_time_intervals(top_path,
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

    for i, indexes in interval_indexes.items():
        new_traj_path = os.path.join(out_path, interval_names[i])

        with mda.Writer(new_traj_path, n_atoms=u.atoms.n_atoms) as W:
            print(f"Creating trajectory for interval {i}")
            print(f"Path: {new_traj_path}")
            print(f"Number of frames: {len(indexes)}")
            for frame in indexes:
                u.trajectory[frame]
                W.write(u)



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
        new_traj_paths.append(new_traj_path)


    top_paths = [top_path]*n_intervals

    return new_traj_paths, top_paths


if __name__ == "__main__":


    badMD_top_path = "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BPTI/BadMD_BPTI_r5_15010_concatenated.pdb"
    badMD_traj_path = [badMD_top_path.replace(".pdb", ".xtc")][0]
    MD_traj_to_time_intervals(badMD_top_path, badMD_traj_path)

    goodMD_top_path = "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BPTI/GoodMD_BPTI_r10_10010_concatenated.pdb"
    goodMD_traj_path = [goodMD_top_path.replace(".pdb", ".xtc")][0]
    MD_traj_to_time_intervals(goodMD_top_path, goodMD_traj_path)



    badMD_top_path = "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BRD4/BadMD_BRD4_r5_15010_concatenated.pdb"
    badMD_traj_path = [badMD_top_path.replace(".pdb", ".xtc")][0]
    MD_traj_to_time_intervals(badMD_top_path, badMD_traj_path)

    goodMD_top_path = "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BRD4/GoodMD_BRD4_r6_12006_concatenated.pdb"
    goodMD_traj_path = [goodMD_top_path.replace(".pdb", ".xtc")][0]
    MD_traj_to_time_intervals(goodMD_top_path, goodMD_traj_path)



    badMD_top_path = "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/HOIP/BadMD_HOIP_r5_15010_concatenated.pdb"
    badMD_traj_path = [badMD_top_path.replace(".pdb", ".xtc")][0]
    MD_traj_to_time_intervals(badMD_top_path, badMD_traj_path)

    goodMD_top_path = "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/HOIP/GoodMD_HOIP_r10_10010_concatenated.pdb"
    goodMD_traj_path = [goodMD_top_path.replace(".pdb", ".xtc")][0]
    MD_traj_to_time_intervals(goodMD_top_path, goodMD_traj_path)



    badMD_top_path = "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/LXRa/BadMD_LXR_r5_15010_concatenated.pdb"
    badMD_traj_path = [badMD_top_path.replace(".pdb", ".xtc")][0]
    MD_traj_to_time_intervals(badMD_top_path, badMD_traj_path)

    goodMD_top_path = "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/LXRa/GoodMD_LXR_r10_10010_concatenated.pdb"
    goodMD_traj_path = [goodMD_top_path.replace(".pdb", ".xtc")][0]
    MD_traj_to_time_intervals(goodMD_top_path, goodMD_traj_path)



    badMD_top_path = "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/MBP/BadMD_MBP_r5_15010_concatenated.pdb"
    badMD_traj_path = [badMD_top_path.replace(".pdb", ".xtc")][0]
    MD_traj_to_time_intervals(badMD_top_path, badMD_traj_path)

    goodMD_top_path = "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/MBP/GoodMD_MBP_r10_10010_concatenated.pdb"
    goodMD_traj_path = [goodMD_top_path.replace(".pdb", ".xtc")][0]
    MD_traj_to_time_intervals(goodMD_top_path, goodMD_traj_path)
