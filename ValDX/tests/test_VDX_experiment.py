import pytest
import os
import MDAnalysis as mda
import subprocess
import pandas as pd
from ValDX.ValidationDX import ValDXer
from ValDX.VDX_Settings import Settings
from ValDX.helpful_funcs import conda_to_env_dict, dfracs_to_df, segs_to_df

name = 'pytest'
calc_name = "test"
MD_rep = "Run_1"
train_frac = 0.5
expt_data_dir = "BPTI_expt_data"
sim_data_dir = "BPTI_simulations"
segs_name = "BPTI_residue_segs.txt"
hdx_name = "BPTI_expt_dfracs.dat"
top_name = "bpti_5pti_eq6_protonly.gro"
traj_name = "bpti_5pti_reimg_protonly.xtc"

rates_name = "BPTI_Intrinsic_rates.dat"


@pytest.fixture(scope="session")
def settings():
    env_path = conda_to_env_dict("HDXER_ENV")  # Assuming HDXER_ENV is the name of your Conda environment
    command = "echo $HDXER_PATH"

    output = subprocess.run(command, shell=True, env=env_path, capture_output=True, text=True, check=True)
    hdxer_path = output.stdout.strip()  # Extract the path from environment variable

    return Settings(hdxer_path=hdxer_path)

def clean_dir(settings, name):
    exp_dir = os.path.join(settings.data_dir, name)

    try:
        os.removedirs(exp_dir)
    except:
        pass

def test_real_initialisation(settings):
    clean_dir(settings, name)

    init_test = ValDXer(settings, name=name)
    assert init_test.name == name

@pytest.fixture(scope="session")
def test_class(settings):
    clean_dir(settings, name)

    settings.data_dir = "ValDX"+os.sep+"test_data"

    return ValDXer(settings, name=name)

def test_load_HDXer(test_class):

    # test_class.load_HDXer()
    # does this only work 
    assert test_class.load_HDXer()


def test_load_HDX_data(test_class):
    times = test_class.settings.times
    data_dir = test_class.settings.data_dir

    expt_path = os.path.join(data_dir, expt_data_dir, hdx_name)
    segs_path = os.path.join(data_dir, expt_data_dir, segs_name)
    print(expt_path)
    print(segs_path)
    hdx, segs = test_class.load_HDX_data(expt_path, segs_path,calc_name=calc_name, experimental=True)

    # select the paths for "calc_name"==calc_name
    paths = test_class.paths[test_class.paths["calc_name"] == calc_name]

    print(paths)

    test_expt_path = paths["HDX"].values[0]
    test_segs_path = paths["SEG"].values[0]

    assert test_expt_path == expt_path
    assert test_segs_path == segs_path

    hdx_test = dfracs_to_df(expt_path, names=times)

    segs_test = segs_to_df(segs_path)

    print(hdx_test)
    print(hdx)

    assert hdx_test.equals(hdx.drop(columns=["calc_name"]))
    assert segs_test.equals(segs.drop(columns=["calc_name"]))


def test_load_structures(test_class):
    data_dir = test_class.settings.data_dir

    top_path = os.path.join(data_dir, sim_data_dir, MD_rep, top_name)
    traj_path = os.path.join(data_dir, sim_data_dir, MD_rep, traj_name)
    print(top_path)
    print(traj_path)

    top = mda.Universe(top_path)
    traj = mda.Universe(top_path, traj_path)

    test_class.load_structures(top_path, [traj_path], calc_name=calc_name)

    paths = test_class.paths[test_class.paths["calc_name"] == calc_name]

    test_top_path = paths["top"].values[0]
    test_traj_path = paths["traj"].values[0]
    print(test_top_path)
    print(test_traj_path)
    assert test_top_path == top_path
    assert test_traj_path == [traj_path]

    test_top, test_traj = test_class.prepare_structures(calc_name=calc_name)

    # test that the topology is the same length
    assert len(test_top.atoms) == len(top.atoms)
    # test that the trajectory is the same length
    assert len(test_traj.trajectory) == len(traj.trajectory)
    

def test_load_intrinsic_rates(test_class):
    data_dir = test_class.settings.data_dir

    rates_path = os.path.join(data_dir, expt_data_dir, rates_name)
    print(rates_path)
    test_class.load_intrinsic_rates(rates_path, calc_name=calc_name)

    paths = test_class.paths[test_class.paths["calc_name"] == calc_name]

    test_rates_path = paths["int_rates"].values[0]

    assert test_rates_path == rates_path

    

def test_split_segments(test_class):
    mode = 'r'
    times = test_class.settings.times
    data_dir = test_class.settings.data_dir
    expt_name = "test_expt"

    test_class.generate_directory_structure
    # load HDX data
    expt_path = os.path.join(data_dir, expt_data_dir, hdx_name)
    segs_path = os.path.join(data_dir, expt_data_dir, segs_name)
    
    test_segs = segs_to_df(segs_path)

    hdx, segs = test_class.load_HDX_data(expt_path, 
                                         segs_path, 
                                         calc_name=expt_name, 
                                         experimental=True)

    # split segments
    test_calc_name, train_rep_name, val_rep_name =  test_class.split_segments(seg_name=expt_name,
                              calc_name=calc_name, 
                              mode=mode,
                              random_seed=1,
                              train_frac=train_frac)
    
    assert test_calc_name == calc_name

    train_segs_name = "_".join(["train",test_class.settings.segs_name[0], calc_name, test_class.settings.segs_name[1]])
    val_segs_name = "_".join(["val",test_class.settings.segs_name[0], calc_name, test_class.settings.segs_name[1]])
                              
    _, train_segs_dir = test_class.generate_directory_structure(calc_name=train_rep_name, gen_only=True)
    _, val_segs_dir = test_class.generate_directory_structure(calc_name=val_rep_name, gen_only=True)

    train_segs_path = os.path.join(train_segs_dir, train_segs_name)
    val_segs_path = os.path.join(val_segs_dir, val_segs_name)

    train_segs = segs_to_df(train_segs_path)
    val_segs = segs_to_df(val_segs_path)

    segs = pd.concat([train_segs, val_segs], ignore_index=True)

    assert segs.equals(test_segs)

    # test that the train and val segments are different

### TODO ###
# will these work if load HDXer doesn't work?
# how do we test live HDXer?
def test_predict_HDX():
    pass

def test_reweight_HDX():
    pass

def test_train_HDX():
    pass

def test_validate_HDX():
    pass

def test_run_VDX():
    pass