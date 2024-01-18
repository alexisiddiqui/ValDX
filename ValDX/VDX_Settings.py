### Python settings for the Val HDX project

import os
import platform
import pandas as pd
import numpy as np


class Settings:
    """Settings for the Val HDX project"""

    def __init__(self, name=None, hdxer_path=None):
        self.times: np.ndarray = [0.167, 1, 10, 120] 

        self.HDXer_env = "HDXER_ENV"
        if hdxer_path is not None:
            self.HDXer_path = hdxer_path
        elif hdxer_path is None:
            self.HDXer_path = os.environ["HDXER_PATH"]

        self.data_dir = 'data'
        self.results_dir = 'results'
        self.plot_dir = 'plots'
        self.logs_dir = 'logs'

        if name is not None:
            self.name = name
        else:
            self.name = "VDX"

        self.random_seed = 42
        self.train_frac = 0.8
        self.replicates = 2
        self.HDX_method = "BestVendruscolo"
        self.logfile_name = ("calc_hdx_", ".log")
        self.segs_name = ("residue_segs_", ".txt")
        self.outname = "out_"
        self.HDXer_mopt = "\"{ 'save_detailed' : True }\""
        self.HDXer_stride = 1
        self.stride = 100

        # reweighting
        self.RW_exponent = -3
        self.RW_basegamma = 10**self.RW_exponent
        self.RW_do_reweighting = True
        self.RW_do_params = False
        self.RW_stepfactor = 0.00001
        self.RW_outprefix = "reweighting_gamma_"
        self.RW_restart_interval = 100
        self.gamma_range = (2,10)



    
            