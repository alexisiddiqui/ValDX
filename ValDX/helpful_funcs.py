# Universal functions
import os
import numpy as np
import pandas as pd
import subprocess
import MDAnalysis as mda
from .reweighting import MaxEnt
from scipy.optimize import curve_fit

import cProfile
import pstats
import io

def conda_to_env_dict(env_name):
    """
    Get the environment variables for a given conda environment.

    Parameters:
    env_name (str): The name of the conda environment to get the variables for.

    Returns:
    dict: A dictionary containing the environment variables for the specified conda environment.
    If the environment is not found, returns None.
    """
    # Run the command 'conda env list' and get the output
    result = subprocess.run(['conda', 'env', 'list'], stdout=subprocess.PIPE)
    
    # Decode result to string and split lines
    envs = result.stdout.decode().splitlines()
    
    # Initialize an empty path
    path = None

    # Iterate through the environments
    for env in envs:
        # Skip lines that don't contain a directory path
        if not env.startswith('#') and env.strip():
            # Split the line into its components
            parts = env.split()
            # The environment name should be the first component, and the path should be the last
            name = parts[0].strip()
            env_path = parts[-1].strip()
            # Check if the environment name matches the one we're looking for
            if name == env_name:
                path = env_path
                break

    # If the path was not set, the environment was not found
    if path is None:
        print(f"Environment '{env_name}' not found.")
        return None
    else:
        print(f"Path to '{env_name}' environment: {path}")

        # Get a copy of the current environment variables
        env_vars = os.environ.copy()
        # Update the PATH to include the bin directory of the conda environment
        env_vars['PATH'] = env_path + os.pathsep + env_vars['PATH']
        
        return env_vars


# times = np.array([0.167, 1.0, 10.0, 120.0]) # Create a NumPy array with time points in minutes

# def read_to_df(file):
#     """Read and create a pandas DataFrame for the given argument.
    
#     Args:
#         file: There are four possible options:
#             'segs' - peptide segments
#             'expt' - experimental HDX deuterated fractions
#             'pred' - calculated HDX deuterated fractions
#             'reweighted' - reweighted HDX deuterated fractions
    
#     Returns:
#         df: A pandas DataFrame containing data for the given argument.
#     """
#     if file == 'segs':
#         # Read and create a pandas DataFrame using a residue segments file
#         df = pd.read_csv(os.path.expandvars('$HDXER_PATH/tutorials/BPTI/BPTI_expt_data/BPTI_residue_segs.txt'),
#                          sep='\s+', header=None, names=['ResStr', 'ResEnd'])
#     elif file == 'expt':
#         # Read and create a pandas DataFrame using an experimental deuterated fractions file
#         df = pd.read_csv(os.path.expandvars('$HDXER_PATH/tutorials/BPTI/BPTI_expt_data/BPTI_expt_dfracs.dat'), 
#                          sep='\s+', skiprows=[0], header=None, usecols=[2, 3, 4, 5], names=times)
#     elif file == 'pred':
#         # Read and create a pandas DataFrame using a computed deuterated fractions file
#         df = pd.read_csv(os.path.expandvars('$HDXER_PATH/tutorials/BPTI/BPTI_calc_hdx/BPTI_SUMMARY_segment_average_fractions.dat'), 
#                          sep='\s+', skiprows=[0], header=None, usecols=[2, 3, 4, 5], names=times)
#     elif file == 'reweighted':
#         df = pd.read_csv(os.path.expandvars('$HDXER_PATH/tutorials/BPTI/BPTI_reweighting/reweighting_gamma_2x10^0_final_segment_fractions.dat'), 
#                          sep='\s+', skiprows=[0], header=None, names=times)
#     else:
#         print("Incorrect argument given. Please choose one of the following: 'segs' 'expt' 'pred' 'reweighted'")
#     return df


def segs_to_df(path: str, names=['ResStr', 'ResEnd']):
    """Read and create a pandas DataFrame using a residue segments file.
    
    Args:
        path: The path to the residue segments file.
    
    Returns:
        df: A pandas DataFrame containing data for the given argument.
        names: A list of column names for the DataFrame. (ResStr, ResEnd)
    """
    df = pd.read_csv(path, sep='\s+', header=None, names=names)
    df["peptide"] = df.index
    return df

def segs_to_file(path: str, df: pd.DataFrame):
    """Write a residue segments file from a pandas DataFrame.
    
    Args:
        path: The path to the residue segments file.
        df: A pandas DataFrame containing data for the given argument.
    """
    df = df.copy()
    # print("segs_to_file")
    # print(df.head())
    # remove peptide column if present make sure we dont overwrite the df in memory.
    if "peptide" in df.columns:
        df = df.drop(columns=["peptide"])
    if "calc_name" in df.columns:
        df = df.drop(columns=["calc_name"])
    if "path" in df.columns:
        df = df.drop(columns=["path"])
    df.to_csv(path, sep='\t', header=False, index=False)

def HDX_to_file(path: str, df: pd.DataFrame):
    """
    Write a HDX deuterated fractions file from a pandas DataFrame.
    Args:
        path: The path to the HDX deuterated fractions file.
        df: A pandas DataFrame containing data for the given argument.
    """
    df = df.copy()
    print(df)
    # remove peptide column if present make sure we dont overwrite the df in memory.
    if "peptide" in df.columns:
        df = df.drop(columns=["peptide"])
    if "calc_name" in df.columns:
        df = df.drop(columns=["calc_name"])
    if "path" in df.columns:
        df = df.drop(columns=["path"])

    header = "\t".join(["#",*[str(col) for col in df.columns]," times/min"])+"\n"
    print(header)
    with open(path, "w") as f:
        f.write(header)
    df.to_csv(path, sep='\t', header=False, index=False, mode="a")


def avgfrac_to_df(path: str, names: list):
    """Read and create a pandas DataFrame using a computed deuterated fractions file.
    
    Args:
        path: The path to the computed deuterated fractions file.
        names: A list of column names for the DataFrame. (times)
    
    Returns:
        df: A pandas DataFrame containing data for the given argument.
    """
    cols = [col+2 for col in range(len(names))]
    df = pd.read_csv(path, sep='\s+', skiprows=[0], header=None, usecols=cols, names=names)
    # this is not true... but 
    df["peptide"] = df.index

    return df

def reweight_to_df(path: str, names: list):
    """Read and create a pandas DataFrame using a reweighted deuterated fractions file.
    
    Args:
        path: The path to the reweighted deuterated fractions file.
        names: A list of column names for the DataFrame. (times)

    
    Returns:
        df: A pandas DataFrame containing data for the given argument.
    """
    df = pd.read_csv(path, sep='\s+', skiprows=[0], header=None, names=names)
    df["peptide"] = df.index
    print(df.shape)
    print(df.head())
    return df


def dfracs_to_df(path: str, names: list):
    print("Path", path)
    df = pd.read_csv(path, sep='\s+', skiprows=[0], header=None)
    #add peptide numbers
    #find number of columns
    ncol = df.shape[1]
    df["peptide"] = df.index

    if ncol == len(names)+2:
        print(f"AVG: ncol = {ncol}, len(names) = {len(names)}")
        return avgfrac_to_df(path, names)
    elif ncol == len(names):
        print(f"RW: ncol = {ncol}, len(names) = {len(names)}")
        return reweight_to_df(path, names)

# def run_MaxEnt(args: tuple[dict, int]):
#     """
#     Run MaxEnt reweighting on HDX data.
#     Takes arg as a dictionary - designed to be used for multiprocessing
#     returns nothing as files are written to disk
#     """
#     args, r = args

#     out_prefix = os.path.join(args["out_prefix"]+f"{r}x10^{args['exponent']}")
#     print(out_prefix)
#     reweight_object = MaxEnt(do_reweight=args["do_reweight"],
#                                 do_params=args["do_params"],
#                                 stepfactor=args["stepfactor"])
    
#     reweight_object.run(gamma=args["basegamma"]*r,
#                         data_folders=args["predictHDX_dir"], 
#                         kint_file=args["kint_file"],
#                         exp_file=args["exp_file"],
#                         times=args["times"], 
#                         restart_interval=args["restart_interval"], 
#                         out_prefix=out_prefix)
    
#     cprofile_log = out_prefix + f"_gamma_{r}x10^{args['exponent']}_cprofile.prof"



def run_MaxEnt(args: tuple[dict, int]):
    """
    Run MaxEnt reweighting on HDX data.
    Takes arg as a dictionary - designed to be used for multiprocessing
    returns nothing as files are written to disk
    """
    pr = cProfile.Profile()
    pr.enable()

    # The original content of run_MaxEnt
    args, r = args
    out_prefix = os.path.join(args["out_prefix"]+f"{r}x10^{args['exponent']}")
    print(out_prefix)
    reweight_object = MaxEnt(do_reweight=args["do_reweight"],
                             do_params=args["do_params"],
                             stepfactor=args["stepfactor"])
    
    (currweights, bv_bc, bv_bh) = reweight_object.run(gamma=args["basegamma"]*r,
                        data_folders=args["predictHDX_dir"], 
                        kint_file=args["kint_file"],
                        exp_file=args["exp_file"],
                        times=args["times"], 
                        restart_interval=args["restart_interval"], 
                        out_prefix=out_prefix)
    
    pr.disable()
    # Save results to a file
    cprofile_log = out_prefix + f"_gamma_{r}x10^{args['exponent']}_cprofile.prof"
    pr.dump_stats(cprofile_log)

    # Print results to the console
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    return (currweights, bv_bc, bv_bh)

def restore_trainval_peptide_nos(calc_name: str, 
                                 expt_name: str,
                                 train_dfs: list[pd.DataFrame],
                                 val_dfs: list[pd.DataFrame],
                                 n_reps: int,
                                 times: list,
                                 train_segs: pd.DataFrame,
                                 val_segs: pd.DataFrame,
                                 expt_segs: pd.DataFrame,
                                 ) -> pd.DataFrame:
    """
    Restores the peptide numbers for the train and validation dataframes for a given calculation and experiment name.
    The function takes in the calculation name, experiment name, train and validation dataframes, number of replicates,
    times, train, validation and experiment segments dataframes. It then iterates through the replicates and adds the 
    correct peptide numbers to the train and validation dataframes. It then merges the replicates together and merges 
    the train and validation dataframes together. Finally, it checks that each replicate has the same number of peptides 
    between the train and validation data and returns the merged dataframe.
    
    Parameters:
    -----------
    calc_name : str
        The name of the calculation.
    expt_name : str
        The name of the experiment.
    train_dfs : list[pd.DataFrame]
        A list of dataframes containing the training data.
    val_dfs : list[pd.DataFrame]
        A list of dataframes containing the validation data.
    n_reps : int
        The number of replicates.
    times : list
        A list of times.
    train_segs : pd.DataFrame
        A dataframe containing the training segments.
    val_segs : pd.DataFrame
        A dataframe containing the validation segments.
    expt_segs : pd.DataFrame
        A dataframe containing the experiment segments.
        
    Returns:
    --------
    merge_df : pd.DataFrame
        A merged dataframe containing the train and validation data.
    """
    # create the replicate names
    train_rep_names = ["_".join(["train", calc_name, str(rep)]) for rep in range(1,n_reps+1)]
    val_rep_names = ["_".join(["val", calc_name, str(rep)]) for rep in range(1,n_reps+1)]

    print("train_rep_names", train_rep_names)
    print("val_rep_names", val_rep_names)
    # iterate through the reps and add the correct peptide numbers to the train and val dfs
    for r in range(n_reps):
        train_rep, val_rep = train_rep_names[r], val_rep_names[r]
                    # 
        train_rep_peptides = train_segs.loc[train_segs["calc_name"] == train_rep, "peptide"].copy().to_list()
        val_rep_peptides = val_segs.loc[val_segs["calc_name"] == val_rep, "peptide"].copy().to_list()

        print("train_rep_peptides", train_rep_peptides)
        print("val_rep_peptides", val_rep_peptides)


        train_dfs[r]["peptide"] = train_rep_peptides
        val_dfs[r]["peptide"] = val_rep_peptides

    # merge the reps together
    train_merge_df = pd.concat(train_dfs, ignore_index=True)
    val_merge_df = pd.concat(val_dfs, ignore_index=True)

    # merge the train and val dfs together
    merge_df = pd.concat([train_merge_df, val_merge_df], ignore_index=True)

    print("manual merge df")
    print(merge_df)

    # make sure that each rep has the same number of peptides between the train and val data
    for r in range(n_reps):
        train_rep_name, val_rep_name = train_rep_names[r], val_rep_names[r]

        train_rep_peptides = train_segs.loc[train_segs["calc_name"] == train_rep_name, "peptide"].values
        val_rep_peptides = val_segs.loc[val_segs["calc_name"] == val_rep_name, "peptide"].values

        rep_peptides = [*train_rep_peptides, *val_rep_peptides]
        rep_peptides = sorted(rep_peptides)
        print("train segs", train_segs.loc[train_segs["calc_name"] == train_rep_name, "peptide"].values)
        print("val segs", val_segs.loc[val_segs["calc_name"] == val_rep_name, "peptide"].values)

        if not np.array_equal(rep_peptides, expt_segs["peptide"].values):
            print("rep_peptides", rep_peptides)
            print("expt_segs", expt_segs["peptide"].values)
            raise ValueError("Peptides not equal between tran/val and expt")

    return merge_df


def add_nan_values(merge_df: pd.DataFrame,
                   calc_name: str,
                   n_reps: int,
                   times: list,
                   expt_segs: pd.DataFrame,
                   ) -> pd.DataFrame:
    """
    Adds nan values to the train and validation dataframes for a given calculation and experiment name.

    Args:
    - merge_df (pd.DataFrame): The dataframe to add nan values to.
    - calc_name (str): The name of the calculation to add nan values for.
    - n_reps (int): The number of replicates to add nan values for.
    - times (list): A list of column names to add nan values for.
    - expt_segs (pd.DataFrame): The dataframe containing the experiment segments.

    Returns:
    - pd.DataFrame: The dataframe with added nan values.
    """

    train_rep_names = ["_".join(["train", calc_name, str(rep)]) for rep in range(1,n_reps+1)]
    val_rep_names = ["_".join(["val", calc_name, str(rep)]) for rep in range(1,n_reps+1)]
    

    empty_df = pd.DataFrame(columns=[*times, "peptide", "calc_name"])
    empty_df["peptide"] = expt_segs["peptide"].values

    nan_df = pd.DataFrame()

    # add nan values to train dfs
    for r in range(n_reps):
        train_rep, val_rep = train_rep_names[r], val_rep_names[r]

        train_df = merge_df[merge_df["calc_name"] == train_rep].copy()
        val_df = merge_df[merge_df["calc_name"] == val_rep].copy()
        # set all values to nan
        # for all values in columns in *times
        for t in times:
            val_df[t] = [np.nan]*len(val_df)

        # switch the calc_name
        val_df["calc_name"] = [train_rep]*len(val_df)

        train_df = pd.concat([train_df, val_df], ignore_index=True)
        nan_df = pd.concat([nan_df, train_df], ignore_index=True)

    for r in range(n_reps):
        train_rep, val_rep = train_rep_names[r], val_rep_names[r]
        # add nan values to val dfs
        train_df = merge_df[merge_df["calc_name"] == train_rep].copy()
        val_df = merge_df[merge_df["calc_name"] == val_rep].copy()
        # set all values to nan
        for t in times:
            train_df[t] = [np.nan]*len(train_df)
        # switch the calc_name
        train_df["calc_name"] = [val_rep]*len(train_df)

        val_df = pd.concat([train_df, val_df], ignore_index=True)
        nan_df = pd.concat([nan_df, val_df], ignore_index=True)

    return nan_df



def calc_LogP_by_res(structure: mda.Universe, B_C=0.35, B_H=2.0, cut_C=6.5, cut_H=2.4):
    # cut_C, cut_H = 6.5, 2.4  # Angstroms

    n_C, n_H = [], []
    structure = structure.select_atoms("protein")
    for res in structure.residues:
        resid = res.resid
        # print("Resi: ", resid)

        amide_N = res.atoms.select_atoms("name N")
        amide_H = res.atoms.select_atoms("name H or name H1 or name H2 or name H3")

        amide_N_pos_string = " ".join([str(i) for i in amide_N.positions[0]])

        heavy_atom_selection = f"point {amide_N_pos_string} {cut_C} and not name H* and not (name N and resid {resid})"
        heavy_atoms = structure.select_atoms(heavy_atom_selection)
        n_C.append(len(heavy_atoms))
        
        amide_H_positions = amide_H.positions
        total = 0
        for hydrogen in amide_H_positions:
            hydrogen_pos_string = " ".join([str(i) for i in hydrogen])
            acceptor_atom_selection = f"point {hydrogen_pos_string} {cut_H} and (type O) and not (name N and resid {resid})"
            acceptor_atoms = structure.select_atoms(acceptor_atom_selection)

            total += len(acceptor_atoms)

        # Handle case where no hydrogens are found
        if len(amide_H_positions) > 0:
            n_H.append(total / len(amide_H_positions))
        else:
            n_H.append(0) # No hydrogens found

    n_C = np.array(n_C)
    n_H = np.array(n_H)

    Log_Pf_C = B_C * n_C
    Log_Pf_H = B_H * n_H
    LogPf_by_res = Log_Pf_C + Log_Pf_H

    return LogPf_by_res


def calc_traj_LogP_byres(universe:mda.Universe, B_C, B_H, stride=1, residues:np.array=None, weights:list=[1]):
    # convert residues to indices
    print("residues", residues)
    seg_indices = np.subtract(residues, 1)
    seg_indices = seg_indices.astype(int)
    print("seg_indices", seg_indices)
    HDX_free_energy = 0
    traj_len = len(universe.trajectory)
    print(traj_len)

    if len(weights) != traj_len:
        weights = [1]*traj_len

    if traj_len > 1:
        for idx, ts in enumerate(universe.trajectory[::stride]):
            LogPf_by_res = calc_LogP_by_res(universe, B_C, B_H)
            print("LogPf_by_res", LogPf_by_res)
            LogPf_by_res = LogPf_by_res[seg_indices]
        
        LogPf_by_res = LogPf_by_res*weights[idx]

        return (LogPf_by_res)/(len(universe.trajectory)/stride)
    elif traj_len == 1:
        LogPf_by_res= calc_LogP_by_res(universe, B_C, B_H)
        LogPf_by_res = LogPf_by_res[seg_indices]

        return (LogPf_by_res)


def kints_to_dict(rates_path):
    rates = pd.read_csv(rates_path, sep="\s+", header=None, skiprows=1)
    rates_dict = rates.set_index(0).to_dict()[1]
    return rates_dict

def merge_kint_dicts_into_df(kint_dicts:list[dict]):
    # key = Resid
    # value = kint
    # convert dicts to dfs
    kint_dfs = []

    for kint_dict in kint_dicts:
        kint_df = pd.DataFrame.from_dict(kint_dict, orient="index")
        kint_df = kint_df.reset_index()
        kint_df.columns = ["Resid", "kint"]
        kint_dfs.append(kint_df)

    # aggregate by resid and mean Kint
    kint_df = pd.concat(kint_dfs, ignore_index=True)
    kint_df = kint_df.groupby("Resid").mean()
    kint_df = kint_df.reset_index()
    kint_df.columns = ["Resid", "kint"]
    return kint_df

def calc_dfrac_uptake_from_LogPf(LogPf_by_res, kints:dict, times:list, residues:list):

    assert len(LogPf_by_res) == len(residues), "LogPf_by_res and kints must be the same length"
    Pf_by_res = np.exp(LogPf_by_res)

    kints = np.array([kints[res] for res in residues])

    kobs_by_res = kints/Pf_by_res

    times = np.array(times).reshape(-1,1)
    dfrac_uptake_by_res = 1 - np.exp(-kobs_by_res*times)

    return dfrac_uptake_by_res
        

