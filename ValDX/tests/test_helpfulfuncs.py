import pytest
import os
import copy
from ValDX.helpful_funcs import *

env_name = "HDXER_ENV" 
times = [0.167, 1, 10, 120]
name = 'pytest'
calc_name = "test"
expt_data_dir = "BPTI_expt_data"
sim_data_dir = "BPTI_simulations"
segs_name = "BPTI_residue_segs.txt"
hdx_name = "BPTI_expt_dfracs.dat"
top_name = "bpti_5pti_eq6_protonly.gro"
traj_name = "bpti_5pti_reimg_protonly.xtc"

rates_name = "BPTI_Intrinsic_rates.dat"

test_dir = os.path.join("ValDX", "test_data")

def test_conda_env_dict():
    env_vars = os.environ.copy()
    var_dict = conda_to_env_dict(env_name)

    print("path")
    print(var_dict["PATH"])
    print("pythonpath")
    print(env_vars["PATH"])

    assert isinstance(var_dict, dict)
    assert env_name in var_dict["PATH"]

def test_segs_to_df():
    
    segs_path = os.path.join(test_dir,expt_data_dir, segs_name)
    test_df = segs_to_df(segs_path)

    assert isinstance(test_df, pd.DataFrame)
    assert 'ResStr' in test_df.columns
    assert 'ResEnd' in test_df.columns
    assert "peptide" in test_df.columns


def test_segs_to_file():
    segs_path = os.path.join(test_dir, expt_data_dir, segs_name)
    test_df = segs_to_df(segs_path)

    test_path = os.path.join(test_dir,expt_data_dir, "test_segs.txt")
    segs_to_file(path=test_path, df=test_df)

    assert os.path.exists(test_path)

    test_df2 = segs_to_df(test_path)

    assert test_df.equals(test_df2)
## need to add seg information to df
# add this to the write df function? we should add checks but add the data seperately
# 
def test_HDX_to_file():
    
    hdx_path = os.path.join(test_dir, expt_data_dir, hdx_name)

    test_df = avgfrac_to_df(hdx_path, times)

    test_path = os.path.join(test_dir, expt_data_dir, "test_hdx.csv")

    HDX_to_file(path=test_path, df=test_df)

    assert os.path.exists(test_path)

    test_df2 = avgfrac_to_df(test_path, times)

    assert test_df.equals(test_df2)


def test_avg_frac_to_file():
    pass

def test_reweight_to_df():
    pass

def test_dfracs_to_df():
    pass

def test_run_MaxEnt():
    pass

def test_restore_peptide_numbers():
    pass

def test_add_nan_values():
    pass
