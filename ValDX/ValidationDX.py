### Class to contain the HDX experiment

# inputs:
# settings
# experimental data
# matched topology and peptide-residue segments
# simulation MD data 

from ValDX.VDX_Settings import Settings
from ValDX.Experiment_ABC import Experiment
import pandas as pd
import time
import os 
import pickle
import glob
import subprocess

class ValDX(Experiment):
    def __init__(self, settings: Settings, name=None):
        super().__init__(settings, name)
        if name is not None:
                self.name = name
        else:
             self.name = self.settings.name

        self.HDX_data = pd.DataFrame()
        
    
    def load_HDX_data(self, HDX_path, SEG_path, calc_name: str=None, experimental=False):
        """
        Load HDX data and Matched Residue Segments: Experiment and Predicted.
        """
        pass

    def load_structures(self, top_path, traj_paths:str=None, calc_name: str=None):
        """
        Load structures: topology and trajectories (Optional).
        If no trajectories are provided, the topology is used to generate a single frame trajectory.
        """
        pass

    def load_HDXer(self):
        """
        Load HDXer: HDXer environment and HDXer executable.
        """
        pass

    def predict_HDX(self, calc_name: str=None):
        """
        Predict HDX data from MD trajectories.
        """
        pass

    ### Dont need to mess with this for now
    def load_intrinsic_rates(self, path, calc_name: str=None):
        """
        Load intrinsic rates: intrinsic rates from HDXer. Optional.
        Otherwise default values are used.
        """
        pass
