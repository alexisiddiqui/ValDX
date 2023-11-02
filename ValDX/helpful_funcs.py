# Universal functions
import numpy as np
import pandas as pd

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
    # remove peptide column if present make sure we dont overwrite the df in memory.
    if "peptide" in df.columns:
        to_write = df.drop(columns=["peptide"])
    to_write.to_csv(path, sep='\s+', header=False, index=False)

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
    return df


def dfracs_to_df(path: str, names: list):

    df = pd.read_csv(path, sep='\s+', skiprows=[0], header=None)
    #add peptide numbers
    df["peptide"] = df.index
    #find number of columns
    ncol = df.shape[1]

    if ncol == len(names)+2:
        return avgfrac_to_df(path, names)
    if ncol == len(names):
        return reweight_to_df(path, names)