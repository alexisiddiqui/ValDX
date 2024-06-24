# script to convert alphafold traj + json into confidence intervals


import os 
import MDAnalysis as mda
import numpy as np
import json



n_intervals = 10
interval_size = 1/n_intervals

confidence_intervals = range(0, n_intervals)
confidence_intervals = [(i/n_intervals, (i/n_intervals + interval_size)) for i in confidence_intervals]
# round to 2 decimal places
confidence_intervals = [(round(lower, 2), round(upper, 2)) for lower, upper in confidence_intervals]

confidence_intervals = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

def af_traj_to_confidence_intervals(top_path,
                                    traj_path,
                                    confidence_intervals,
                                    json_path=None,
                                    min_interval_size=500,
                                    out_path=None):
    
    if json_path is None:
        json_path = top_path.replace("protonated.pdb", "ranks.json")

    # load the trajectory
    u = mda.Universe(top_path, traj_path)

    if out_path is None:
        out_path = os.path.join(os.path.dirname(traj_path), 'af_confidence_intervals')

    os.makedirs(out_path, exist_ok=True)

    traj_name = os.path.basename(traj_path)

    # load the json file
    with open(json_path, 'r') as f:
        data = json.load(f)

    keys = list(data.keys())
    # sort keys by x.split('_')[1]
    keys.sort(key=lambda x: int(x.split('_')[1]))

    # extract plddt into a 1d array
    plddt = np.array([data[key][str(i)]['plddt'] for key in keys for i in (data[key].keys())])

    # for each confidence interval, extract the frames that fall within the percentile intervals
    # and save them to a new trajectory
    print(confidence_intervals)

    # extract the frame indexes first
    frame_indexes = {str(interval[0])+"_"+str(interval[1]): [] for interval in confidence_intervals}
    for i, interval in enumerate(confidence_intervals):
        print(f"Extracting frames for interval {interval}")
        lower, upper = interval
        lower_plddt = np.percentile(plddt, lower*100)
        upper_plddt = np.percentile(plddt, upper*100)
        frame_indexes[str(interval[0])+"_"+str(interval[1])] = np.where((plddt >= lower_plddt) & (plddt <= upper_plddt))[0].tolist()

    # check if the intervals have enough frames
    for key in frame_indexes.keys():
        if len(frame_indexes[key]) < min_interval_size:
            print(f"Interval {key} has less than {min_interval_size} frames. Decrease the number of intervals or the minimum interval size")
            raise ValueError
        

    # create a new trajectory for each interval
    for key in frame_indexes.keys():
        print(f"Creating trajectory for interval {key}")
        print(f"Number of frames: {len(frame_indexes[key])}")
        new_traj_path = os.path.join(out_path, traj_name.replace(".xtc", f"_{key}.xtc"))

        with mda.Writer(new_traj_path, n_atoms=u.atoms.n_atoms) as W:
            for frame in frame_indexes[key]:
                u.trajectory[frame]
                W.write(u)

    # create trajectories for the top and bottom confidence intervals (min_interval_size frames)
    # find the top min_interval_size plddt frames
    top_frames = np.argsort(plddt)[-min_interval_size:]
    bottom_frames = np.argsort(plddt)[:min_interval_size]

    top_traj_path = os.path.join(out_path, traj_name.replace(".xtc", f"_top_{str(min_interval_size)}.xtc"))
    bottom_traj_path = os.path.join(out_path, traj_name.replace(".xtc", f"_bottom_{str(min_interval_size)}.xtc"))

    with mda.Writer(top_traj_path, n_atoms=u.atoms.n_atoms) as W:
        print(f"Creating trajectory for top {min_interval_size} frames")
        for frame in top_frames:
            u.trajectory[frame]
            W.write(u)

    with mda.Writer(bottom_traj_path, n_atoms=u.atoms.n_atoms) as W:
        print(f"Creating trajectory for bottom {min_interval_size} frames")
        for frame in bottom_frames:
            u.trajectory[frame]
            W.write(u)



if __name__ == "__main__":
    top_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10000_protonated.pdb"
    traj_paths = ["/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10000_protonated.xtc"][0]

    af_traj_to_confidence_intervals(top_path, traj_paths, confidence_intervals)

    top_path = "/home/alexi/Documents/ValDX/raw_data/BRD4/BRD4_APO/BRD4_APO_484_1_af_sample_127_10000_protonated.pdb"
    traj_paths = ["/home/alexi/Documents/ValDX/raw_data/BRD4/BRD4_APO/BRD4_APO_484_1_af_sample_127_10000_protonated.xtc"][0]

    af_traj_to_confidence_intervals(top_path, traj_paths, confidence_intervals)

    top_path = "/home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo/HOIP_apo697_1_af_sample_127_10000_protonated.pdb"
    traj_paths = ["/home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo/HOIP_apo697_1_af_sample_127_10000_protonated.xtc"][0]

    af_traj_to_confidence_intervals(top_path, traj_paths, confidence_intervals)

    top_path = "/home/alexi/Documents/ValDX/raw_data/LXRalpha/LXRalpha_APO/LXRa200_1_af_sample_127_10000_protonated.pdb"
    traj_paths = ["/home/alexi/Documents/ValDX/raw_data/LXRalpha/LXRalpha_APO/LXRa200_1_af_sample_127_10000_protonated.xtc"][0]

    af_traj_to_confidence_intervals(top_path, traj_paths, confidence_intervals)



    top_path = "/home/alexi/Documents/ValDX/raw_data/MBP/MBP_wt_1_af_sample_127_10000_protonated.pdb"
    traj_paths = ["/home/alexi/Documents/ValDX/raw_data/MBP/MBP_wt_1_af_sample_127_10000_protonated.xtc"][0]

    af_traj_to_confidence_intervals(top_path, traj_paths, confidence_intervals)
