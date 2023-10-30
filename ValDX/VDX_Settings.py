### Python settings for the Val HDX project

import os
import platform
import pandas as pd
import numpy as np


class Settings:
    """Settings for the Val HDX project"""

    def __init__(self, name=None):
        self.times: np.ndarray = [0.167,10,120] 

        self.HDXer_env = "HDXER_ENV"
        self.HDXer_path = os.environ["HDXER_PATH"]


        if name is not None:
            self.name = name
        else:
            self.name = "VDX"
            