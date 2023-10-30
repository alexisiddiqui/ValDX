### C

from abc import ABC, abstractmethod
from ValDX.VDX_Settings import Settings
import pandas as pd
import MDAnalysis as mda

class Experiment(ABC):
    def __init__(self,settings: Settings, name=None):
        super().__init__(settings, name)
        self.settings = settings

        if name is not None:
            self.name = name
        else:
            self.name = "ABC"
        self.calc_names = []
        self.paths = pd.DataFrame()
        self.structures = pd.DataFrame()

    def prepare_structures(self, calc_name: str=None):
        """
        Prepares MDA Universe object from topology and trajectory files.
        """
        pass

    def generate_directory_structure(self, calc_name: str=None, overwrite=False):
        """
        Generates directory structure for the experiment. 
        This is for the outputs of the experiment.
        """
        pass

    @abstractmethod
    def prepare_config(self):
        """
        Prepares the configuration...for the environment setup.
        Includes HDXER env as well as the HDXER executable.
        """
        pass

    @abstractmethod
    def save_experiment(self):
        pass

    @abstractmethod
    def load_experiment(self):
        pass