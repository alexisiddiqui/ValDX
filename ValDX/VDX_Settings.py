### Python settings for the Val HDX project

import os
import platform
import pandas as pd
import numpy as np


class Settings:
    """Settings for the Val HDX project"""

    def __init__(self, name=None):
        self.times: np.ndarray = [0.167, 1, 10, 120] 

        self.HDXer_env = "HDXER_ENV"
        self.HDXer_path = os.environ["HDXER_PATH"]

        self.data_dir = 'data'
        self.results_dir = 'results'
        self.plot_dir = 'plots'

        if name is not None:
            self.name = name
        else:
            self.name = "VDX"

        self.random_seed = 42
        self.train_frac = 0.8
        self.replicates = 5
        self.HDX_method = "BestVendruscolo"
        self.logfile_name = ("calc_hdx_", ".log")
        self.segs_name = ("residue_segs_", ".txt")
        self.outname = "out_"
        self.HDXer_mopt = "{ 'save_detailed' : True }"
    
            