import pytest
import os
from ValDX.Experiment_ABC import Experiment
from ValDX.VDX_Settings import Settings

settings = Settings(hdxer_path="this is a test")
name = 'pytest'

def test_initialisation():

    exp_dir = os.path.join(settings.data_dir, name)

    try:
        os.removedirs(exp_dir)
    except:
        pass

    init_class = Experiment(settings, name="pytest")
    assert init_class.name == "pytest"
    init_class = Experiment(settings)
    assert init_class.name == "ABC"
    assert init_class.settings == settings

def test_generate_directory_structure():
    test_class = Experiment(settings,name=name)

    exp_dir = os.path.join(settings.data_dir, name)

    try:
        os.removedirs(exp_dir)
    except:
        pass

    test_calc_name, test_exp_dir = test_class.generate_directory_structure(overwrite=True)

    assert test_exp_dir == exp_dir
    assert test_calc_name == name

    test_calc_name, test_exp_dir = test_class.generate_directory_structure(overwrite=False)

    assert test_exp_dir != exp_dir
    assert test_calc_name != name

def test_generate_directory_structure_calc_name():
    calc_name = "test"
    exp_dir = os.path.join(settings.data_dir, name, calc_name)
    test_exp_dir = os.path.join(settings.data_dir, name, calc_name)
    test_class = Experiment(settings,name=name)

    try:
        os.rmdir(exp_dir)
    except:
        pass

    test_calc_name, test_exp_dir = test_class.generate_directory_structure(overwrite=True, calc_name=calc_name)

    assert test_exp_dir == exp_dir
    assert test_calc_name == calc_name

    test_calc_name, test_exp_dir = test_class.generate_directory_structure(calc_name=calc_name, gen_only=True)

    assert test_exp_dir == exp_dir
    assert test_calc_name == calc_name

def test_save_and_load():

    calc_name = "test"
    test_class = Experiment(settings,name=name)

    save_path = test_class.save_experiment(save_name=calc_name)

    assert os.path.exists(save_path)

    test_class2 = Experiment(settings,name=name)

    test_class2.load_experiment(load_path=save_path)

    assert test_class2.name == name
    assert test_class2.settings == test_class.settings

    assert isinstance(test_class, Experiment)
    assert isinstance(test_class2, Experiment)