# Universal functions
import os
import numpy as np
import pandas as pd
import subprocess
from HDXer.reweighting import MaxEnt

def conda_to_env_dict(env_name):
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

def run_MaxEnt(args: tuple[dict, int]):
    """
    Run MaxEnt reweighting on HDX data.
    Takes arg as a dictionary - designed to be used for multiprocessing
    returns nothing as files are written to disk
    """
    args, r = args

    out_prefix = os.path.join(args["out_prefix"]+f"{r}x10^{args['exponent']}")
    print(out_prefix)
    reweight_object = MaxEnt(do_reweight=args["do_reweight"],
                                do_params=args["do_params"],
                                stepfactor=args["stepfactor"])
    
    reweight_object.run(gamma=args["basegamma"]*r,
                        data_folders=args["predictHDX_dir"], 
                        kint_file=args["kint_file"],
                        exp_file=args["exp_file"],
                        times=args["times"], 
                        restart_interval=args["restart_interval"], 
                        out_prefix=out_prefix)