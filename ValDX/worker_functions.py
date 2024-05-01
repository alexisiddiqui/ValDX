from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import os



def process_mode(mode, settings, hdx_path, segs_path, expt_name, top_path, traj_paths, weights, random_seeds, data_dir):
    from ValDX.ValidationDX import ValDXer
    print(f"Running {mode} split mode")
    settings.split_mode = mode
    _VDX = ValDXer(settings=settings)
    _VDX.settings.plot = False
    _VDX.load_HDX_data(HDX_path=hdx_path, SEG_path=segs_path, calc_name=expt_name)
    _VDX.load_structures(top_path=top_path, traj_paths=traj_paths, calc_name=expt_name)
    _VDX.run_VDX(calc_name=expt_name, weights=weights, expt_name=expt_name, random_seeds=random_seeds)
    _, _, name = _VDX.dump_analysis()
    return _VDX.analysis_data, name
