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
 
        self.load_HDXer()
        self.generate_directory_structure(overwrite=False)
    
    def load_HDX_data(self, HDX_path=None, SEG_path=None, calc_name: str=None, experimental=False):
        """
        Load HDX data and Matched Residue Segments paths: Experiment and Predicted.
        Does not check if the paths are valid. 
        May fail if incorrect format during prepare HDX data.
        """
        if calc_name is None:
            raise ValueError("Please provide a calculation name for the structures.")
        if HDX_path is None and experimental is False:
            raise ValueError(f"Please provide a path to the experimental {calc_name} HDX data.")

        if HDX_path is None:
            paths_to_add = pd.DataFrame({"SEG": [SEG_path], "calc_name": [calc_name], "experimental": [experimental]})
        elif SEG_path is None:
            paths_to_add = pd.DataFrame({"HDX": [HDX_path], "calc_name": [calc_name], "experimental": [experimental]})
        elif HDX_path and SEG_path is not None:
            paths_to_add = pd.DataFrame({"HDX": [HDX_path], "SEG": [SEG_path], "calc_name": [calc_name], "experimental": [experimental]})
        self.paths = pd.concat([self.paths, paths_to_add], ignore_index=True)

        hdx, segs  = self.prepare_HDX_data(calc_name=calc_name)

        if hdx and segs is None:
            raise ValueError(f"Unable to prepare any HDX data for {calc_name}.")

    def load_structures(self, top_path, traj_paths:str=None, calc_name: str=None):
        """
        Load structures: topology and trajectories (Optional).
        If no trajectories are provided, the topology is used to generate a single frame trajectory.
        """
        if calc_name is None:
            raise ValueError("Please provide a calculation name for the structures.")

        if traj_paths is None:
            traj_paths = top_path

        paths_to_add = pd.DataFrame({"top": [top_path], "traj": [traj_paths], "calc_name": [calc_name]})
        self.paths = pd.concat([self.paths, paths_to_add], ignore_index=True)

        top, traj = self.prepare_structures(calc_name=calc_name)

    def load_HDXer(self, HDXer_env=None, HDXer_path=None):
        """
        Load HDXer: HDXer environment and HDXer executable.
        """

        if HDXer_env is not None:
            self.HDXer_env = HDXer_env
            self.HDXer_path = os.environ["HDXER_PATH"]
        elif HDXer_path is not None:
            self.HDXer_path = HDXer_path

        test_HDX_command = ['python', self.HDXer_path, '-h']

        try:
            subprocess.run(test_HDX_command, env=self.HDXer_env, check=True)
            return True
        except subprocess.CalledProcessError:
            print("HDXer failed to run. Check the HDXer environment and executable path.")
            return False


    def predict_HDX(self, calc_name: str=None, mode: str=None, rep: int=None, train: bool=True):
        """
        Predict HDX data from MD trajectories.
        """
        if calc_name is None:
            raise ValueError("Please provide a calculation name for the structures.")
        rep_name = "_".join([calc_name, mode, str(rep)])        
        out_dir = os.path.join(self.settings.data_dir, self.name, calc_name)

        calc_hdx = os.path.join(self.HDXer_path, "HDXer", "calc_hdx.py")
        top = self.paths.loc[self.paths["calc_name"] == calc_name, "top"]
        trajs = self.paths.loc[self.paths["calc_name"] == calc_name, "traj"]
        log = os.path.join(out_dir, self.settings.logfile_name[0] + calc_name + self.settings.logfile_name[1])
        if train:
            segs = self.train_segs.loc[self.train_segs["calc_name"] == calc_name, "path"]
        else:
            segs = self.val_segs.loc[self.val_segs["calc_name"] == calc_name, "path"]
        out_prefix = os.path.join(out_dir, "_".join([self.settings.outname, rep_name]))

        if self.load_HDXer():

            calc_hdx_command = ["python",
                                calc_hdx,
                                "-t", trajs,
                                "-p", top,
                                "-m", self.settings.HDX_method,
                                "-log", log,
                                "-out", out_prefix,
                                "-segs", segs,
                                "-mopt", self.settings.HDXer_mopt]
                                

            subprocess.run(calc_hdx_command, env=self.HDXer_env, check=True)

            ### add HDX data back into class

    def reweight_HDX(self, calc_name: str=None, mode: str=None, rep: int=None, train: bool=True):
        # how do we validate the reweighting? we can compare to experimental
        # for validation we would reweight and then compare to experimental too?
        pass

    def evaluate_HDX(self, calc_name: str=None, mode: str=None, n_reps: int=None):
        # compare both training and validation data to the experimental data
        # show the averages and the distributions of the errors
        pass

    ### Dont need to mess with this for now
    def load_intrinsic_rates(self, path, calc_name: str=None):
        """
        Load intrinsic rates: intrinsic rates from HDXer. Optional.
        Otherwise default values are used.
        """
        pass
